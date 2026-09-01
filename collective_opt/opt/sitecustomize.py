"""Load the collective optimization when this directory is on PYTHONPATH."""

from __future__ import annotations

import os
import sys
from pathlib import Path


if os.environ.get("SGLANG_ENABLE_QUANTIZED_CP_MOE_AG", "0") == "1":
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from opt.runtime_patch import install_import_hook

    install_import_hook()
