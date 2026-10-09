import unittest
import tempfile
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments
from safetensors.torch import load_file

from pard.pard2_train import TargetFeatEmbedWarp, Pard2Trainer, WarpDataCollator


class PilotLossTest(unittest.TestCase):
    def test_cat_uses_prior_reference_probabilities(self):
        collator = WarpDataCollator(None, para_num=4, unused_tokenids=[31] * 3)
        probabilities = torch.tensor([[1.0, 0.2, 0.3, 0.4, 0.5]])
        weights = collator._build_prev_prob(probabilities, 4).reshape(4, 5)
        # At the last reference token, positions 1–4 depend on 0–3 prior tokens.
        torch.testing.assert_close(
            weights[:, -1], torch.tensor([1.0, 0.4, 0.3 * 0.4, 0.2 * 0.3 * 0.4]),
        )

    def test_disabled_features_match_feature_free_forward(self):
        torch.manual_seed(42)
        student = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        ))
        model = TargetFeatEmbedWarp(student, target_dim=24, target_feat_mask=1.0).eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            dropped = model(input_ids=ids, labels=ids, target_feat=torch.randn(1, 4, 24))
            absent = model(input_ids=ids, labels=ids)
        torch.testing.assert_close(dropped.logits, absent.logits, rtol=0, atol=0)

    def test_distillation_with_different_padded_head_sizes(self):
        torch.manual_seed(42)
        student = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=40, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        ))
        model = TargetFeatEmbedWarp(
            student, target_dim=24, target_head=torch.nn.Linear(24, 48, bias=False),
            target_feat_mask=0.5,
            distill_vocab_size=32,
        )
        ids = torch.tensor([[1, 2, 3, 4]])
        output = model(
            input_ids=ids, labels=ids, prev_prob=torch.ones_like(ids).float(),
            target_logits=torch.randn(1, 4, 24), target_feat=torch.randn(1, 4, 24),
        )
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertGreater(student.model.embed_tokens.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.target_head.weight.grad)

    def test_trainer_saves_tied_student_weights(self):
        student = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            tie_word_embeddings=True,
        ))
        model = TargetFeatEmbedWarp(student, target_dim=24)
        with tempfile.TemporaryDirectory() as directory:
            # Exercise the Trainer saving path without SFT dataset preparation.
            trainer = Pard2Trainer.__new__(Pard2Trainer)
            Trainer.__init__(trainer, model=model, args=TrainingArguments(
                output_dir=directory, report_to="none", use_cpu=True,
            ))
            trainer.save_model()
            state = load_file(str(Path(directory) / "model.safetensors"))
            model.load_state_dict(state, strict=True)
            self.assertTrue(torch.equal(
                state["base_model.lm_head.weight"], state["base_model.model.embed_tokens.weight"],
            ))


if __name__ == "__main__":
    unittest.main()
