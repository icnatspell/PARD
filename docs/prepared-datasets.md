# Prepared dataset contract

The PARD-2 trainer consumes local JSON/JSONL files listed in
`general.data_path`. For externally prepared data, each row provides a `text`
string containing the complete chat-formatted training sequence:

```json
{"text":"<|im_start|>user\nWhat is two plus two?<|im_end|>\n<|im_start|>assistant\nFour.<|im_end|>\n"}
```

Use the chosen instruct tokenizer's chat template during external preparation,
including the intended system prompt, all context turns, assistant answer, and
end-of-turn tokens. The trainer consumes this text directly and tokenizes it;
it does not regenerate answers or apply another chat template. A `data` template
section is unnecessary for this format. Token-array ingestion is not implemented.

The teacher scores these existing sequences during training to supply reference
probabilities, hidden states, and distillation targets. Teacher-generated answers
can be used, but generation is external and optional for this input contract.

The legacy `conversation` format remains supported for single prompt/answer
pairs. Use prepared `text` for multi-turn conversations: the legacy formatter
reads only the first pair.

## Supervision and length

For the Qwen pilot, `collator.end_token_id: 151644` selects the final
`<|im_start|>` segment. Earlier turns remain context; loss includes the final
assistant segment's role/header and response, rather than every assistant turn.
Prepared samples must end with an assistant answer. If every assistant turn
should receive supervision, prepare separate prefixes ending at each answer
externally, or change the collator in a subsequent experiment.

Choose `train.max_seq_length` for the actual corpus. The local smoke value of
64 tokens is only for the short fixtures; using it for the full corpus would
truncate most examples. Truncation must retain the intended answer segment.
Keep training and evaluation data separate during external preparation.

## Local smoke test

The prepared pilot includes one single-turn and one multi-turn formatted
fixture and deliberately has no chat-template configuration:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/check_prepared_dataset.py --config config/train/pilot_prepared_qwen25_instruct.yaml
HF_HUB_OFFLINE=1 .venv/bin/python -m pard.pard2_train -c config/train/pilot_prepared_qwen25_instruct.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/check_feature_free_draft.py datas/ckpt/Qwen2.5-0.5B-Instruct_prepared_text_pilot
```

Run the preflight on the full corpus before training. It reports token-length
percentiles and the number of truncated samples, and fails on malformed text
rows, a final non-assistant turn, or truncation that removes every final-answer
token. It is a Qwen JSONL check for the current final-segment supervision policy.
It does not check dataset deduplication or benchmark overlap.

On the full-training PC, copy/adapt a configuration and point its `data_path`
at the owner's prepared corpus. Select appropriate sequence lengths, training
budget, optimizer, and teacher placement for that machine. The two-step pilot
is not the full-training recipe.

Evaluate resulting checkpoints using the owner's NVIDIA SpeedBench setup.
No SpeedBench adapter or benchmark run is part of the local smoke test.

## Verified locally — 2026-10-09

The actual 0.5B-Instruct student and 7B-Instruct teacher completed two steps
on the prepared fixtures with finite CE/KD losses and nonzero gradients.
Checkpoint conversion and feature-free four-token drafting after reload passed.
Peak PyTorch GPU allocation was 5909.8 MiB. The single-turn and multi-turn
fixtures contain 33 and 47 tokens respectively, retaining the final answer
within the pilot's 64-token limit. See [the result report](../reports/prepared-dataset-smoke.json).
