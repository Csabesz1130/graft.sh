"""graft.toml: where the genome lives, which models exist, and how a rebase runs."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FILE = "graft.toml"
RUNGS = ("zero-shot", "prompt", "lora")


class ConfigError(Exception):
    pass


@dataclass
class Project:
    root: Path
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def genome_dir(self) -> Path:
        return self.root / str(self.raw.get("genome", {}).get("path", "genome"))

    @property
    def holdout(self) -> float:
        value = float(self.raw.get("genome", {}).get("holdout", 0.2))
        if not 0.0 <= value < 1.0:
            raise ConfigError(f"[genome] holdout must be at least 0 and below 1, got {value}")
        return value

    @property
    def state_dir(self) -> Path:
        return self.root / ".graft"

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        models = self.raw.get("models", {})
        if not isinstance(models, dict):
            raise ConfigError("[models] must be a table of tables, one per model")
        return models

    def model(self, name: str) -> dict[str, Any]:
        if name not in self.models:
            known = ", ".join(sorted(self.models)) or "none"
            raise ConfigError(f"no model named {name!r} in {FILE} (defined: {known})")
        return self.models[name]

    @property
    def rebase(self) -> dict[str, Any]:
        return self.raw.get("rebase", {})

    @property
    def trainer(self) -> dict[str, Any]:
        return self.raw.get("trainer", {})

    @property
    def suites(self) -> dict[str, dict[str, Any]]:
        return self.raw.get("suites", {})

    def ladder(self, override: list[str] | None = None) -> list[str]:
        ladder = override or self.rebase.get("ladder") or ["zero-shot", "prompt"]
        if isinstance(ladder, str):
            ladder = [part.strip() for part in ladder.split(",") if part.strip()]
        unknown = [rung for rung in ladder if rung not in RUNGS]
        if unknown or not ladder:
            raise ConfigError(f"ladder rungs are {', '.join(RUNGS)}; got {', '.join(unknown) or 'nothing'}")
        return sorted(set(ladder), key=RUNGS.index)


def find_root(start: Path | None = None) -> Path:
    """The nearest folder at or above `start` that holds graft.toml."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / FILE).is_file():
            return folder
    raise ConfigError(f"no {FILE} here or in any parent folder; run `graft init` first")


def load(start: Path | None = None) -> Project:
    root = find_root(start)
    try:
        raw = tomllib.loads((root / FILE).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(f"{root / FILE}: {err}") from err
    return Project(root=root, raw=raw)


TEMPLATE = """\
# graft.toml: the genome, the models, and how a rebase runs. See docs/user-guide.md.

[genome]
path = "genome"
holdout = 0.2            # share of cases never shown to the new model while it is adapted

# The model that has the behavior today. Any OpenAI-compatible endpoint works.
# [models.prod]
# endpoint = "http://127.0.0.1:8443/v1"
# model = "acme/support-v3"
# api_key_env = "GRAFT_API_KEY"

# The new base model to rebase onto.
# [models.new-base]
# endpoint = "http://127.0.0.1:8000/v1"
# model = "org/new-base-30b"

[rebase]
# source = "prod"
ladder = ["zero-shot", "prompt"]   # add "lora" once a [trainer] is set up
rounds = 3
"""
