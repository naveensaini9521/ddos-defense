"""Load scenario YAML and drive traffic_generator."""
# TODO: implement
"""
simulator/scenarios.py

Runs named scenarios by loading YAML configs from simulator/configs/.
Emits labels to data/labeled/labels.jsonl.

Usage:
    python -m simulator.scenarios --config simulator/configs/http_flood.yaml \
        --target http://192.168.100.10/
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from core.logging import get_logger
from simulator.traffic_generator import TrafficGenerator

log = get_logger("simulator.scenarios")


def load_scenario(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = {"name", "duration"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"scenario missing keys: {missing}")
    return cfg


def run_scenario(cfg: dict, target: str) -> None:
    name = cfg["name"]
    duration = int(cfg.get("duration", 30))
    gen = TrafficGenerator(target=target)

    log.info(f"running scenario={name} duration={duration}s target={target}")

    try:
        if name == "normal":
            gen.run_normal(duration=duration, rps=cfg.get("rps", 5.0))

        elif name == "http_flood":
            gen.run_http_flood(
                duration=duration,
                threads=cfg.get("threads", 50),
                spoof_xff=bool(cfg.get("headers", {}).get("X-Forwarded-For") == "rotate"),
            )

        elif name == "slowloris":
            gen.run_slowloris(
                duration=duration,
                connections=cfg.get("connections", 200),
                keepalive=cfg.get("keepalive_seconds", 10.0),
            )

        elif name == "syn_flood":
            # L4 — needs raw sockets; delegated to a separate tool
            # (see simulator/raw_syn.py if you want Scapy)
            log.warning("syn_flood requires root + scapy — use a separate script")
            time.sleep(duration)

        else:
            raise ValueError(f"unknown scenario: {name}")

    except KeyboardInterrupt:
        log.info("scenario interrupted")
        gen.stop()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to scenario YAML")
    p.add_argument("--target", required=True, help="http://host:port/")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_scenario(args.config)
    run_scenario(cfg, args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())