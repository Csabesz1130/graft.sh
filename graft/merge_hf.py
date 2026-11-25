"""Fold a graft's adapter into its base: one plain Hugging Face model folder, for engines that take no adapters
(GGUF conversion for llama.cpp, Goethe).

    python -m graft.merge_hf --base <weights or hub id> --adapter <folder> --output <folder>

Without --adapter it writes the base as it is, which is how a hub id becomes a folder to convert. An adapter folder
that holds a whole model (the trainer's --method full) is copied. Needs `pip install 'graft-sh[train]'`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m graft.merge_hf", description=__doc__.split("\n\n")[0])
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args(argv)
    if args.output.exists():
        p.error(f"{args.output} already exists")
    if args.adapter and (args.adapter / "config.json").is_file() and not (args.adapter /
                                                                           "adapter_config.json").is_file():
        shutil.copytree(args.adapter, args.output)   # a fully trained model: nothing to merge
        print(f"copied the trained model {args.adapter} to {args.output}")
        return 0
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("merging needs torch and transformers: pip install 'graft-sh[train]'", file=sys.stderr)
        return 1
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, low_cpu_mem_usage=True)
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError:
            print("merging an adapter needs peft: pip install 'graft-sh[train]'", file=sys.stderr)
            return 1
        model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(json.dumps({"merged": str(args.output), "base": args.base,
                      "adapter": str(args.adapter) if args.adapter else None, "dtype": args.dtype}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
