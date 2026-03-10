#!/usr/bin/env python3
"""Launch diffusion multiruns through Hydra.

Example:
  python scripts/orchestrator.py network.wrapper.timesteps=5,10,15 data.split=sim70,sim80
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
        "network/wrapper=diffusion",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)
