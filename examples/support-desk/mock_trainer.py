#!/usr/bin/env python3
"""A stand-in for a trainer: reads a graft dataset and writes an "adapter" that knows the rules it saw.

    python3 mock_trainer.py <dataset.jsonl> <output folder>

A real trainer has the same contract: read the dataset, write the adapter into the output folder,
exit 0. See docs/user-guide.md#training.
"""
import json
import sys
from pathlib import Path

dataset, output = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
rules = sorted({row["rule"] for row in rows if row.get("rule")})   # replay rows have no rule
output.mkdir(parents=True, exist_ok=True)
(output / "learned.json").write_text(json.dumps({"rules": rules, "rows": len(rows)}, indent=2) + "\n")
print(f"read {len(rows)} rows covering {len(rules)} rules; wrote {output}")
