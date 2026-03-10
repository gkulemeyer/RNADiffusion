#!/usr/bin/env python3
"""Launch supervised multiruns through Hydra.

Example:
  python scripts/orchestrator_supervised.py data.split=sim60,sim70 +fold_number=0,1,2
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    run_script = Path(__file__).resolve().parent / "run.py"
    cmd = [
        sys.executable,
        str(run_script),
        "-m",
        "pipeline=train",
        "network/wrapper=supervised",
        "network/backbone=simpleunet",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)
