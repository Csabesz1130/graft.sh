"""The built-in trainer: fit a LoRA adapter (or, for small models, all weights) to a graft dataset.

    python -m graft.train_hf --dataset dataset.jsonl --base <weights> --output <folder>

Each dataset line is {"messages": [..., assistant], "tools": [...], "weight": n}. Only the final assistant
message is learned; everything before it is context. Needs `pip install 'graft-sh[train]'`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from graft.runners import dtype_argument, hf_messages


def encode(tokenizer: Any, row: dict[str, Any], max_length: int) -> tuple[list[int], list[int]] | None:
    """Token ids and labels for one row; the labels cover the final assistant message only."""
    messages, tools = hf_messages(row["messages"]), row.get("tools") or None
    whole = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    before = tokenizer.apply_chat_template(messages[:-1], tools=tools, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(whole, add_special_tokens=False)["input_ids"]
    if len(ids) > max_length:
        return None
    context = len(tokenizer(before, add_special_tokens=False)["input_ids"]) if whole.startswith(before) else 0
    return ids, [-100] * context + ids[context:]


def targets(value: str) -> str | list[str]:
    """--targets as peft takes it: the string all-linear, or a list of module names."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    return "all-linear" if names in ([], ["all-linear"]) else names


def distillation_loss(torch: Any, model: Any, x: Any, y: Any) -> Any:
    """KL(base || adapted) over the reply's tokens: the adapter must keep the base model's whole next-token
    distribution there, not just its single sampled reply. The base is the same weights with the adapter off."""
    model.eval()   # the base's own view: no dropout
    with torch.no_grad(), model.disable_adapter():
        teacher = model(input_ids=x).logits[:, :-1].float()
    model.train()
    student = model(input_ids=x).logits[:, :-1].float()
    mask = (y[:, 1:] != -100)
    log_t, log_s = torch.log_softmax(teacher, -1), torch.log_softmax(student, -1)
    per_token = (log_t.exp() * (log_t - log_s)).sum(-1)
    return (per_token * mask).sum() / mask.sum().clamp(min=1)


def copies(weight: float) -> list[float]:
    """How a row's weight is trained: one full copy per whole unit, and a fraction as one copy whose loss is
    scaled by it. 2.5 is two copies and a half; 0.5 is one copy at half strength; 0 or less counts once."""
    weight = float(weight)
    if weight <= 0:
        return [1.0]
    whole, part = int(weight), weight - int(weight)
    return [1.0] * whole + ([part] if part > 1e-9 else [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m graft.train_hf", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--base", required=True, help="local folder or Hugging Face id of the new base model")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=("lora", "full"), default="lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8, help="rows per optimizer step (gradients are accumulated)")
    parser.add_argument("--max-length", type=int, default=8192, help="rows longer than this many tokens are skipped")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-loss", choices=("kl", "ce"), default="kl",
                        help="for replay rows: kl matches the base model's whole next-token distribution (the "
                             "adapter switched off) on the reply; ce trains on the reply's tokens only. "
                             "A full fine-tune has no base to compare with and uses ce")
    parser.add_argument("--targets", default="all-linear",
                        help="the layers a LoRA adapts: all-linear, or module names separated by commas "
                             "(on a hybrid model, attention and MLP layers only keeps the state-space layers intact)")
    args = parser.parse_args(argv)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("the built-in trainer needs torch, transformers and peft: pip install 'graft-sh[train]'", file=sys.stderr)
        return 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    cuda = torch.cuda.is_available()
    kwargs: dict[str, Any] = {dtype_argument(): torch.bfloat16 if cuda else torch.float32}
    if cuda:
        try:
            import accelerate  # noqa: F401
            kwargs["device_map"] = "auto"
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_pretrained(args.base, **kwargs)
    if cuda and "device_map" not in kwargs:
        model = model.to("cuda")
    if args.method == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                                                 target_modules=targets(args.targets), task_type="CAUSAL_LM"))
    device = next(model.parameters()).device

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples, skipped = [], 0
    for row in rows:
        encoded = encode(tokenizer, row, args.max_length)
        if encoded is None:
            skipped += 1
            continue
        replay = row.get("origin") == "replay"
        examples += [(*encoded, scale, replay) for scale in copies(row.get("weight", 1))]
    if not examples:
        print(f"no row fits in {args.max_length} tokens", file=sys.stderr)
        return 1
    print(f"{len(rows)} rows, {len(examples)} after weights, {skipped} skipped as too long; "
          f"method {args.method}, {args.epochs} epochs, lr {args.lr}, device {device}", flush=True)

    distill = args.replay_loss == "kl" and args.method == "lora"
    if args.replay_loss == "kl" and args.method != "lora" and any(e[3] for e in examples):
        print("a full fine-tune has no base model to distil from: replay rows use --replay-loss ce", flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    model.train()
    step = 0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        total = 0.0
        for index, (ids, labels, scale, replay) in enumerate(examples, 1):
            x, y = torch.tensor([ids], device=device), torch.tensor([labels], device=device)
            if replay and distill:
                loss = distillation_loss(torch, model, x, y)
            else:
                loss = model(input_ids=x, labels=y).loss
            (loss * scale / args.batch_size).backward()
            total += float(loss)
            if index % args.batch_size == 0 or index == len(examples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
        print(f"epoch {epoch}/{args.epochs}  loss {total / len(examples):.4f}  steps {step}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved {args.method} weights to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
