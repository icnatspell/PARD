# Custom PARD training

## Scope

Train Qwen2.5-0.5B-Instruct with an unquantized Qwen2.5-7B-Instruct teacher,
then deploy the quantized drafter without verifier hidden states. The initial
deployment verifier accepts at most four draft candidates.

PARD consumes prepared datasets. Source selection, deduplication, chat formatting,
and optional teacher-response generation are managed externally. Do not couple
the trainer to individual dataset sources or add a response-generation pipeline
as a prerequisite for training.

This PC is for plumbing and smoke tests. The prepared training corpus resides
on the full-training machine. NVIDIA SpeedBench is the owner's intended
evaluation benchmark; connecting the owner's harness and measuring acceptance
and deployment throughput are separate from local smoke tests.

The owner wants to compare the custom PARD-2 checkpoint against the official
`amd/PARD-Qwen2.5-0.5B` checkpoint, using the same Qwen2.5-7B-Instruct verifier
and SPEED-Bench prompts. Both drafters must run without verifier features.
See [the comparison protocol](docs/acceptance-comparison.md).

## Owner's prepared mix

Counts and licenses below are supplied by the owner. Counts are before final
cross-source deduplication; the final corpus contains 93,470 samples.

| Source | License | Role | Samples |
| --- | --- | --- | ---: |
| UltraChat 200k | MIT | Chat | 35,000 |
| UltraFeedback | MIT | Chat | 10,000 |
| AutoIF-instruct-61k | Apache-2.0 | Instruction | 10,000 |
| UltraInteract SFT | MIT | Reasoning | 10,000 |
| MetaMathQA | MIT | Math | 4,000 |
| Orca-Math | MIT | Math | 3,000 |
| Evol-CodeAlpaca-v1 | Apache-2.0 | Code | 3,000 |
| OpenBallast ballast-t2 | CC BY 4.0 | RAG | 17,000 |
| Aya Dataset | Apache-2.0 | Multilingual | 10,000 |

See [the prepared dataset contract](docs/prepared-datasets.md) and
[local pilot results](docs/instruct-pilot.md).
