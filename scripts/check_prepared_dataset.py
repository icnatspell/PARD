"""Check prepared Qwen JSONL text boundaries and configured truncation."""

import argparse
import json
import math
from pathlib import Path

import yaml
from transformers import AutoTokenizer


def check_sequence(ids, assistant_header, start_id, end_id, max_length):
    starts = [i for i, token in enumerate(ids) if token == start_id]
    if not starts:
        return "missing chat turn boundary"
    start = starts[-1]
    if ids[start:start + len(assistant_header)] != assistant_header:
        return "final turn is not an assistant answer"
    body_start = start + len(assistant_header)
    retained = ids[:max_length]
    if not any(token != end_id for token in retained[body_start:]):
        return "truncation leaves no final assistant answer tokens"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    general = config["general"]
    max_length = config["train"]["max_seq_length"]
    if max_length <= 0:
        parser.error("max_seq_length must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        general["model_name"], cache_dir=general.get("cache_dir", "cache"),
    )
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    if start_id is None or end_id is None or header[0] != start_id:
        parser.error("this preflight supports Qwen chat-formatted data")
    if config.get("collator", {}).get("end_token_id") != start_id:
        parser.error("collator.end_token_id must select the Qwen turn boundary")

    lengths, errors = [], []
    error_count = rows = truncated = 0
    paths = general["data_path"]
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        with Path(path).open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                        raise ValueError("expected a JSON object with a text string")
                    ids = tokenizer.encode(row["text"], add_special_tokens=False)
                    lengths.append(len(ids))
                    truncated += int(len(ids) > max_length)
                    error = check_sequence(ids, header, start_id, end_id, max_length)
                except ValueError as exc:
                    error = str(exc)
                if error:
                    error_count += 1
                    if len(errors) < 20:
                        errors.append({"file": str(path), "line": line_number, "reason": error})
    lengths.sort()
    if not rows:
        error_count += 1
        errors.append({"reason": "dataset contains no rows"})
    report = {
        "rows": rows, "max_seq_length": max_length, "truncated_rows": truncated,
        "token_lengths": {
            "min": lengths[0], "p50": lengths[math.ceil(len(lengths) * 0.5) - 1],
            "p95": lengths[math.ceil(len(lengths) * 0.95) - 1], "max": lengths[-1],
        } if lengths else None,
        "error_count": error_count, "first_errors": errors,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(1 if error_count else 0)


if __name__ == "__main__":
    main()
