#!/usr/bin/env bash
set -euo pipefail
python -m ml.train --data data/labeled --out ml/models/v1.pkl
