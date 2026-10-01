#!/usr/bin/env bash
set -euo pipefail
echo "[bootstrap] installing python deps..."
pip install -r requirements.txt
echo "[bootstrap] done."
