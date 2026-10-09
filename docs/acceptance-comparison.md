# PARD versus custom PARD-2 acceptance comparison

Run the official `amd/PARD-Qwen2.5-0.5B` checkpoint and the custom trained
PARD-2 student against the same `Qwen/Qwen2.5-7B-Instruct` verifier. Use
SPEED-Bench prompts and report both aggregate and per-category results.
Both drafters must run without verifier hidden states.

[NVIDIA's AIPerf guide](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/profile-with-speed-bench-dataset)
describes SPEED-Bench profiling and server-side acceptance collection. Prompts
can also be fed to a standalone evaluator; acceptance counters must come from
the speculative verification loop rather than a text-similarity comparison.

## Controlled runs

1. Qwen2.5-7B-Instruct target-only generation for correctness and timing reference.
2. The official PARD checkpoint, unmodified.
3. The custom PARD-2 student with feature injection absent.
4. Optionally, ordinary PARD trained on the same prepared corpus and budget,
   to separate workload adaptation from PARD-2's training additions.

Begin with greedy decoding and four draft candidates, then sweep 1–4.
Keep verifier revision, precision, prompt formatting, prompt selection, context
and output limits, concurrency, and runtime settings fixed. Record drafter
revisions/artifact hashes and the SPEED-Bench dataset version. Check output
tokens against target-only greedy decoding under the same stopping rules.

For every round, record the number of proposed candidates and the length of
the consecutively accepted draft prefix. Stop acceptance at the first mismatch
with the verifier's greedy decision. Report:

- Accepted draft-prefix length per round (0–4 for a four-token block).
- Acceptance fraction: total accepted draft tokens / total proposed draft tokens.
- Prefix survival by position: fraction of eligible rounds accepting at least
  1, 2, 3, or 4 draft tokens.
- Committed tokens per round, separating verifier correction/bonus tokens.
- End-to-end committed tokens/second and latency, with execution breakdowns.

Handle EOS/output-budget tails explicitly: count only candidates actually
proposed under the common stopping policy, and keep correction/bonus tokens out
of draft acceptance metrics. Aggregate from total counts; report category results
separately rather than silently averaging incompatible ratios.

## Integration work still required

The existing Qwen evaluator already pairs the official PARD checkpoint with
Qwen2.5-7B-Instruct, but reads benchmark-specific `data` rows. SPEED-Bench input
format/category handling needs an adapter or the owner's existing harness.
If using AIPerf, the serving runtime must support each feature-free parallel
drafter and expose the required counters; this has not been validated here.

The existing PARD evaluators' `accept_length` includes the verifier-emitted token,
and their `acc` averages only positions checked before the first rejection.
Those fields are not directly the draft-only metrics defined above. Add explicit
counters before comparing results, and verify feature-free checkpoint loading:
the PARD-2 evaluator otherwise auto-enables feature injection from its config.

Unquantized GPU runs measure model agreement and GPU performance. Repeat the
comparison after drafter quantization against the fixed deployed quantized
7B-Instruct verifier to establish NPU acceptance and deployment throughput.
No acceptance benchmark or SPEED-Bench integration has been executed locally.
