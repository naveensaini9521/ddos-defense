#!/usr/bin/env bash
set -euo pipefail
SCENARIO="${1:-http_flood}"
python -m simulator.scenarios --config "simulator/configs/${SCENARIO}.yaml"
