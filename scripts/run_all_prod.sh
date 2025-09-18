#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
PYTHONPATH=. python scripts/run_report.py
