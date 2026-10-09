"""Compare packed PARD training with independently constructed causal prefixes."""

import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from pard.pard2_train import TargetFeatEmbedWarp, WarpDataCollator


class ParallelAlignmentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        backend = Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="0"))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="0")
        self.ids = torch.arange(1, 10).unsqueeze(0)
        self.length = self.ids.shape[1]
        # The teacher server supplies features shifted by one token already.
        self.features = torch.arange(self.length * 8).reshape(1, self.length, 8).float()
        self.features[:, 0] = 0
        self.hidden = self.features + 100
        self.probabilities = torch.linspace(0.2, 0.9, self.length).unsqueeze(0)
        student = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            attn_implementation="eager",
        ))
        self.model = TargetFeatEmbedWarp(
            student, target_dim=8, target_feat_mask=0.0,
        ).eval()

    def collate(self, ratio=1.0):
        collator = WarpDataCollator(
            self.tokenizer, use_target_model=True, para_num=4,
            unused_tokenids=[63] * 3, end_token_id=None,
            down_sample_ration=ratio, down_sample_ration_min=0.2,
        )
        with patch.object(collator, "_get_client") as client:
            client.return_value.get_target_feat.return_value = (
                self.features, self.hidden, self.probabilities,
            )
            return collator([{"input_ids": self.ids[0].tolist()}])

    def check_packed(self, batch, origins):
        with torch.no_grad():
            packed = self.model(**batch).logits
            # Labels/weights are consumed one position after their logits.
            for row, origin in enumerate(origins[:-1]):
                horizon, position = divmod(origin, self.length)
                anchor = position - horizon
                if anchor < 0 or position == self.length - 1:
                    continue
                prefix = self.ids[:, :anchor + 1]
                tokens = torch.cat([prefix, torch.full((1, horizon), 63)], dim=1)
                teacher_features = torch.cat([
                    self.features[:, :anchor + 1],
                    self.features[:, anchor:anchor + 1].repeat(1, horizon, 1),
                ], dim=1)
                independent = self.model(
                    input_ids=tokens, labels=tokens, target_feat=teacher_features,
                    use_cache=False,
                ).logits[:, -1]
                torch.testing.assert_close(packed[:, row], independent, rtol=1e-5, atol=1e-6)
                self.assertEqual(batch["labels"][0, row + 1], self.ids[0, position + 1])
                expected_weight = self.probabilities[0, anchor + 1:position + 1].prod()
                torch.testing.assert_close(batch["prev_prob"][0, row + 1], expected_weight)
                torch.testing.assert_close(batch["target_feat"][:, row], self.features[:, anchor])
                torch.testing.assert_close(batch["target_logits"][:, row], self.hidden[:, position])

    def test_four_horizons_match_independent_prefixes(self):
        batch = self.collate()
        self.check_packed(batch, list(range(4 * self.length)))
        mask = batch["attention_mask"][0, 0]
        for horizon in range(4):
            for position in range(horizon, self.length):
                anchor = position - horizon
                allowed = set(range(anchor + 1))
                allowed.update(block * self.length + anchor + block for block in range(1, horizon + 1))
                actual = set((mask[horizon * self.length + position] == 0).nonzero().flatten().tolist())
                self.assertEqual(actual, allowed)

    def test_conditional_dropping_preserves_predictions_and_targets(self):
        batch = self.collate(ratio=0.7)
        positions = batch["position_ids"][0].tolist()
        # Teacher-hidden markers independently recover each retained original row.
        original_positions = ((batch["target_logits"][0, :, 0] - 100) / 8).long().tolist()
        horizons = []
        horizon = 0
        for i, position in enumerate(positions):
            if i and position <= positions[i - 1]:
                horizon += 1
            horizons.append(horizon)
        origins = [h * self.length + p for h, p in zip(horizons, original_positions)]
        self.assertLess(len(origins), 4 * self.length)
        self.assertEqual(horizons[-1], 3)
        self.check_packed(batch, origins)


if __name__ == "__main__":
    unittest.main()
