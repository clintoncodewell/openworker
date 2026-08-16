"""The council: one question, every configured model, a real debate, one finding."""

from __future__ import annotations

from .config import (
    DEFAULT_ROLES,
    PRESETS,
    CouncilConfig,
    Source,
    config_path,
    load_config,
    save_config,
)
from .core import default_panel, make_council_tool, run_council
from .scratchpad import Scratchpad, runs_dir
from .sources import KINDS as SOURCE_KINDS

__all__ = [
    "CouncilConfig",
    "DEFAULT_ROLES",
    "PRESETS",
    "SOURCE_KINDS",
    "Scratchpad",
    "Source",
    "config_path",
    "default_panel",
    "load_config",
    "make_council_tool",
    "run_council",
    "runs_dir",
    "save_config",
]
