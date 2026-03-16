#!/usr/bin/env bash
set -euo pipefail

conda run --no-capture-output -n seq2seq python scripts/run_archiveii_sim_sweep.py "$@"
