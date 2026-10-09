# Qwen2.5 instruct PARD-2 pilot

Use Qwen/Qwen2.5-0.5B-Instruct as the student and
Qwen/Qwen2.5-7B-Instruct as the frozen, unquantized teacher. The AMD PARD
checkpoint advertises Qwen2.5 architecture and placeholder ID 151665, but
its model card does not establish instruct initialization. Keep it as a
separate warm-start experiment until that provenance is established.

## Local feasibility

The current machine has a 10 GiB RTX 3080 and approximately 23 GiB system RAM.
The teacher alone needs approximately 14–15 GiB for BF16 weights. The pilot
therefore places the teacher on CPU and trains the student on GPU. CPU teacher
scoring will be slow; the two-step run is for infrastructure validation.
Teacher loading and output-head extraction use BF16 and low-memory loading
to avoid a full FP32 teacher allocation. RAM and GPU peaks still need measurement.
For sustained training, use a larger GPU or implement offline teacher-signal
shards; the current collator does not consume precomputed signals.

## Run

Create a dedicated environment with the pilot requirements. The lock file
records the complete environment used locally. DeepSpeed is unnecessary for
this single-device entry point. Do not reuse a Transformers 5 environment for
this Transformers 4.51.3 / TRL 0.14.0 trainer. Run from the repository root:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-pilot.lock.txt
.venv/bin/python -m pard.pard2_train -c config/train/pilot_pard2_qwen25_instruct.yaml
.venv/bin/python scripts/check_feature_free_draft.py datas/ckpt/Qwen2.5-0.5B-Instruct_pard2_instruct_pilot
```

This direct entry point avoids pard.train's default eight-device DeepSpeed
launcher. The config uses four prediction positions, no conditional token
dropping initially, CAT, distillation, and 50% per-example feature dropout.
One final teacher layer is the initial feature source; multi-layer features
can be evaluated later.

The pilot uses Adafactor to reduce optimizer memory while retaining FP32 student
weights and BF16 autocast. It is a two-step infrastructure run, not a tuned
training recipe. Record optimizer choice in subsequent matched experiments.

The tokenizers share 151665 contiguous token IDs, but their padded heads have
151936 (student) and 152064 (teacher) outputs. KD renormalizes both distributions
over the actual shared tokenizer IDs; this excludes unused rows. CAT continues
to use the teacher's reference-token probability from its full output head.
This distinction is part of the training recipe, not a runtime sampling change.

The two short fixture examples are plumbing input only, not a workload or acceptance
experiment. Confirm the rendered template matches the tokenizer chat template
and verify vocabulary IDs before running. Short examples must retain assistant
tokens after truncation to have a useful loss.

## Completion checks

1. Finite CE/KD losses and nonzero student gradients on a teacher-scored batch.
2. Correct response masking, future-position labels, CAT weights, and feature
   alignment, including the conditional-drop path before enabling it.
3. Save/reload the checkpoint and run a four-token proposal with feature injection
   fully absent. Random training dropout alone does not validate deployment.
4. Export the student-only inference path and quantize using the intended NPU
   toolchain. The teacher projection must not become a required runtime input.

Then supply the externally prepared corpus described in
[the dataset contract](prepared-datasets.md), with disjoint evaluation data.
Optional teacher-response generation stays outside this repository's trainer.
Train ordinary PARD and
PARD-2 from the same instruct initialization, data, and comparable budgets.
Evaluate both after quantization against the fixed deployed 7B-Instruct verifier,
with draft lengths 1–4. Use committed tokens per second as the decision metric.

Regression checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests exercise distillation with different padded head sizes and Trainer
checkpoint save/reload with tied student embeddings. A tiny synthetic-model
integration run also completed two training steps, checkpoint conversion, and
a feature-free four-position forward. These checks do not establish workload
acceptance or NPU compatibility; those require the deployed verifier/toolchain.

Additional local correctness checks now compare all four packed prediction
positions with independently constructed causal prefixes, including teacher
features, targets, CAT weights, and conditional token dropping. All 11 tests
pass. Prepared-text boundary/truncation checks also pass on both fixtures;
see `reports/prepared-dataset-preflight.json`.

## Full-model smoke result — 2026-10-09

The actual 0.5B-Instruct / 7B-Instruct pilot completed successfully. Results and
model revisions are recorded in [the smoke report](../reports/instruct-pilot-smoke.json).

| Step | CE | KD | Combined loss | Gradient norm before clipping |
| --- | --- | --- | --- | --- |
| 1 | 10.5689 | 9.7341 | 10.7910 | 441.0154 |
| 2 | 9.2109 | 8.3244 | 9.2455 | 368.0564 |

Total process time was 135.37 seconds, including setup and checkpoint saving.
Peak PyTorch GPU allocation was 5908.5 MiB (about 5.8 GiB). Both checkpoint-2
and the final checkpoint converted successfully. The final student reloaded
in BF16 and proposed four tokens with no teacher features; its first proposal
also matched an ordinary causal forward on the same committed prefix. The
placeholder embedding changed by up to 0.0002151, confirming training updated
the saved student.

Final inference artifact:
`datas/ckpt/Qwen2.5-0.5B-Instruct_pard2_instruct_pilot/pard_model/`.
Full training logs and the feature-free reload result are in `logs/`.
The generated artifacts, model cache, environment, and logs are ignored by Git.

This run validates the training and feature-free checkpoint path only. The two
fixture examples are not a useful quality benchmark. Controlled PARD/PARD-2
training on the owner's prepared corpus, SpeedBench evaluation, and NPU
export/quantization remain the next experiments.

For resumable HTTP model downloads, set `HF_HUB_DISABLE_XET=1` when downloading
with this environment. Xet's partially assembled temporary files from the
earlier interrupted transfer were not reusable across download processes.
