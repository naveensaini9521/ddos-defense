"""Main pipeline runner — the defense loop.

Ties together:
    - pipeline/live_collector.py  (tails logs)
    - collector/aggregator.py     (per-IP features)
    - ml/ensemble.py              (3-model voting)
    - pipeline/dynamic_ip.py      (rotating/subnet attacks)
    - blocker/block_manager.py    (enforcement)

Usage:
    python3 -m pipeline.runner              # run forever
    python3 -m pipeline.runner --once       # one tick
    python3 -m pipeline.runner --dry-run    # no real blocks
    python3 -m pipeline.runner --duration 60
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass, field
from monitoring.metrics import wire_runner

from blocker.block_manager import BlockManager
from collector.aggregator import aggregate
from core.config_loader import load as load_config
from core.logging import get_logger
from pipeline.dynamic_ip import DynamicConfig, DynamicIPDetector
from pipeline.live_collector import LiveCollector, collector_from_config

log = get_logger("pipeline.runner")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class RunnerStats:
    ticks: int = 0
    total_records: int = 0
    total_ips: int = 0
    total_decisions: int = 0
    total_per_ip: int = 0
    total_subnet: int = 0
    total_asn: int = 0
    total_blocks: int = 0
    total_errors: int = 0
    started_at: float = field(default_factory=time.time)

    def summary(self) -> dict:
        return {
            "ticks": self.ticks,
            "total_records": self.total_records,
            "total_ips": self.total_ips,
            "total_decisions": self.total_decisions,
            "total_per_ip": self.total_per_ip,
            "total_subnet": self.total_subnet,
            "total_asn": self.total_asn,
            "total_blocks": self.total_blocks,
            "total_errors": self.total_errors,
            "uptime_seconds": int(time.time() - self.started_at),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    """The main defense loop."""

    def __init__(
        self,
        dry_run: bool = False,
        config_path: str | None = None,
        start_collector: bool = True,
    ) -> None:
        self.cfg = load_config(config_path) if config_path else load_config()
        self.stats = RunnerStats()

        # Read config
        p = self.cfg.get("pipeline", {})
        self.window_seconds = float(p.get("window_seconds", 10.0))

        # Collector
        self.collector = collector_from_config(self.cfg)
        if start_collector:
            self.collector.start()

        # Per-IP detector (ensemble of 3 models)
        self.detector = self._build_detector()

        # Dynamic IP detector (cross-IP / subnet / ASN)
        dyn_cfg = p.get("dynamic_ip", {})
        self.dynamic_enabled = bool(dyn_cfg.get("enabled", True))
        self.dynamic_detector = DynamicIPDetector(DynamicConfig(
            subnet_size=int(dyn_cfg.get("subnet_size", 24)),
            new_ip_rate_threshold=float(dyn_cfg.get("new_ip_threshold", 5.0)),
            window_seconds=self.window_seconds,
        ))

        # Blocker
        self.manager = BlockManager(
            config_path=config_path,
            start_daemon=True,
        )
        if dry_run:
            self.manager.firewall.cfg.dry_run = True
            log.warning("DRY-RUN mode — decisions will be logged but not enforced")
        else:
            dry_run_cfg = self.cfg.get("blocker", {}).get("dry_run", False)
            if dry_run_cfg:
                log.warning("blocker.dry_run=true in config — no real enforcement")

        log.info(f"Runner ready: window={self.window_seconds}s")
        log.info(f"  collector: {self.collector.source_str}")
        log.info(f"  detector:  ensemble of {len(self.detector.detectors)} models")
        log.info(f"  dynamic:   enabled={self.dynamic_enabled} "
                 f"subnet=/{self.dynamic_detector.cfg.subnet_size}")
        log.info(f"  blocker:   backend={self.manager.firewall.cfg.backend} "
                 f"dry_run={self.manager.firewall.cfg.dry_run} "
                 f"use_ipset={self.manager.firewall.cfg.use_ipset}")

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------
    def _build_detector(self):
        from ml.ensemble import EnsembleDetector

        m = self.cfg.get("ml", {})
        ensemble_cfg = m.get("ensemble", {})

        model_paths = {
            "iso": ensemble_cfg.get("iso_path", "ml/models/v11_iso.pkl"),
            "rf":  ensemble_cfg.get("rf_path",  "ml/models/v11_rf.pkl"),
            "xgb": ensemble_cfg.get("xgb_path", "ml/models/v11_xgb.pkl"),
        }
        weights = m.get("weights", {})
        threshold = float(m.get("threshold", 0.7))

        return EnsembleDetector(
            model_paths=model_paths,
            weights=weights or None,
            threshold=threshold,
        )

    # ------------------------------------------------------------------
    # One tick
    # ------------------------------------------------------------------
    def tick(self) -> dict:
        """Run one iteration of the pipeline. Returns stats dict."""
        t0 = time.time()
        self.stats.ticks += 1
        tick_num = self.stats.ticks

        # 1. Records from the last window
        records = self.collector.window(seconds=self.window_seconds)
        self.stats.total_records += len(records)

        if not records:
            log.info(f"tick {tick_num}: no records in last "
                     f"{self.window_seconds:.0f}s")
            return {"tick": tick_num, "records": 0, "ips": 0,
                    "per_ip": 0, "subnet": 0, "blocks": 0}

        # 2. Aggregate per-IP features
        features = aggregate(records)
        self.stats.total_ips += len(features)

        if not features:
            log.info(f"tick {tick_num}: {len(records)} records "
                     f"but no features extracted")
            return {"tick": tick_num, "records": len(records), "ips": 0,
                    "per_ip": 0, "subnet": 0, "blocks": 0}

        # 3b. Dynamic (subnet/ASN) decisions FIRST
        subnet_decisions: list = []
        if self.dynamic_enabled:
            try:
                subnet_decisions = self.dynamic_detector.check(records, features)
            except Exception as e:
                log.warning(f"dynamic detector error: {e}")
        self.stats.total_subnet += len(subnet_decisions)

        # 3a. Per-IP decisions — SKIPPED if we already have a broader block.
        # Rationale: if a subnet or ASN block covers the attack, blocking
        # hundreds of individual IPs from the same range is redundant and
        # floods the firewall with rules.
        if subnet_decisions:
            per_ip_decisions: list = []
            log.info(f"tick {tick_num}: subnet/ASN block active — "
                     f"skipping {len(features)} per-IP decisions")
        else:
            per_ip_decisions = self.detector.predict_many(features)
            self.stats.total_per_ip += len(per_ip_decisions)

        all_decisions = per_ip_decisions + subnet_decisions
        self.stats.total_decisions += len(all_decisions)
        self.stats.total_asn += sum(
            1 for d in subnet_decisions if getattr(d, "scope", "") == "asn"
        )

        # 4. Enforce
        blocks = 0
        from monitoring.metrics import metrics as _m
        for d in all_decisions:
            scope = getattr(d, "scope", "ip")
            source = "subnet" if scope in ("subnet", "asn") else "per_ip"
            try:
                _m.observe_decision(d.ip, d.action, d.confidence, source=source)
            except Exception:
                pass

            if d.action == "block":
                try:
                    result = self.manager.enforce(d)
                    if result.get("action") == "blocked":
                        blocks += 1
                except Exception as e:
                    self.stats.total_errors += 1
                    log.error(f"enforce failed for {d.ip}: {e}")
                    
        self.stats.total_blocks += blocks

        # 5. Summary
        elapsed = time.time() - t0
        log.info(
            f"tick {tick_num}: "
            f"records={len(records)} "
            f"ips={len(features)} "
            f"per_ip={len(per_ip_decisions)} "
            f"subnet={len(subnet_decisions)} "
            f"blocks={blocks} "
            f"({elapsed*1000:.0f}ms)"
        )

        return {
            "tick": tick_num,
            "records": len(records),
            "ips": len(features),
            "per_ip": len(per_ip_decisions),
            "subnet": len(subnet_decisions),
            "decisions": len(all_decisions),
            "blocks": blocks,
            "elapsed_ms": round(elapsed * 1000, 1),
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, duration: float | None = None) -> None:
        """Run the loop until interrupted or duration elapsed."""
        log.info(f"starting main loop "
                 f"(window={self.window_seconds}s, "
                 f"duration={duration or 'forever'})")

        deadline = time.time() + duration if duration else None

        try:
            while True:
                try:
                    self.tick()
                except Exception as e:
                    self.stats.total_errors += 1
                    log.error(f"tick failed: {e}", exc_info=True)

                if deadline and time.time() >= deadline:
                    log.info("duration reached")
                    break

                time.sleep(self.window_seconds)

        except KeyboardInterrupt:
            log.info("interrupted by user")

        finally:
            self.shutdown()

    def run_once(self) -> None:
        """Run one tick and exit."""
        self.tick()
        self.shutdown()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        log.info("shutting down...")
        try:
            self.collector.stop()
        except Exception as e:
            log.warning(f"collector stop failed: {e}")
        try:
            self.manager.shutdown()
        except Exception as e:
            log.warning(f"manager shutdown failed: {e}")

        log.info(f"final stats: {self.stats.summary()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Main DDoS defense loop")
    p.add_argument("--once", action="store_true",
                   help="run one tick and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="do not execute real iptables commands")
    p.add_argument("--duration", type=float, default=None,
                   help="stop after N seconds (default: run forever)")
    p.add_argument("--config", default=None,
                   help="path to config YAML (default: config/config.yaml)")
    p.add_argument("--no-collector", action="store_true",
                   help="do not start the log tailer (test only)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    runner = Runner(
        dry_run=args.dry_run,
        config_path=args.config,
        start_collector=not args.no_collector,
    )
    
    wire_runner(runner)

    def _handle_signal(sig, frame):
        log.info(f"received signal {sig}")
        runner.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if args.once:
        runner.run_once()
    else:
        runner.run(duration=args.duration)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())