"""Reload a converted pilot checkpoint and propose four tokens without features."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Training output containing pard_model/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint / "pard_model", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    assert not any("target_proj" in key or "target_head" in key for key in model.state_dict())
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is two plus two?"}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(args.device)
    inputs = torch.cat([
        prefix, torch.full((1, 3), model.config.pard_token, device=args.device),
    ], dim=1)
    with torch.inference_mode():
        scores = model(inputs, use_cache=False).logits[:, -4:]
        assert torch.isfinite(scores).all()
        # Future placeholders must not change the prediction at the prefix end.
        reference = model(prefix, use_cache=False).logits[:, -1:]
        assert torch.equal(scores[:, :1].argmax(-1), reference.argmax(-1))
        proposals = scores.argmax(-1).cpu().tolist()[0]
    print(json.dumps({
        "checkpoint": str(args.checkpoint), "teacher_features": False,
        "block_size": 4, "proposal_ids": proposals,
        "proposal_text": tokenizer.decode(proposals),
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if args.device.startswith("cuda") else None,
    }, indent=2))


if __name__ == "__main__":
    main()
