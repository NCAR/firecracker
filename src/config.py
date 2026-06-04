"""
config.py

Loads TOML configuration for the Firecracker simulation.
The default config lives at <repo_root>/cfg/default.toml.
Pass an explicit path to load_config() to use an alternative file.
"""

import tomllib
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent.parent / "cfg" / "default.toml"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else _DEFAULT_PATH
    with open(p, "rb") as f:
        return tomllib.load(f)
