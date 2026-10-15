"""Prometheus metrics for the DDoS defense system.

Exposes counters, gauges, and histograms via the prometheus_client library.
Call metrics.observe_decision() and metrics.observe_tick() from the pipeline.

Also provides an HTTP /metrics endpoint (start it separately).

Usage:
    from monitoring.metrics import metrics

    metrics.observe_decision(ip, action, confidence, source)
    metrics.observe_tick(records, latency_seconds)
    metrics.set_active_blocks(15)

Start scrape endpoint:
    python3 -m monitoring.metrics --port 9100
"""
from __future__ import annotations

import threading

from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Info,
    push_to_gateway,
    start_http_server,
)

from core.logging import get_logger

log = get_logger("monitoring.metrics")

# Pushgateway endpoint (edit if running elsewhere)
import os

PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "pushgateway:9091")
PUSHGATEWAY_JOB = os.getenv("PUSHGATEWAY_JOB", "ddos-runner")


# ---------------------------------------------------------------------------
# Metrics definitions
# ---------------------------------------------------------------------------

class DDoSMetrics:
    """Container for all Prometheus metrics."""

    def __init__(self) -> None:
        # --- Counters ---
        self.decisions_total = Counter(
            "ddos_decisions_total",
            "Total decisions made",
            ["action", "source"],
        )
        self.blocks_total = Counter(
            "ddos_blocks_total",
            "Total blocks enforced",
            ["model", "strike_range"],
        )
        self.whitelist_skips_total = Counter(
            "ddos_whitelist_skips_total",
            "Decisions skipped due to whitelist",
        )
        self.errors_total = Counter(
            "ddos_errors_total",
            "Errors in pipeline",
            ["component"],
        )

        # --- Gauges ---
        self.active_blocks = Gauge(
            "ddos_active_blocks",
            "Currently active blocks",
        )
        self.permanent_blocks = Gauge(
            "ddos_permanent_blocks",
            "Permanent block count",
        )
        self.buffer_size = Gauge(
            "ddos_collector_buffer_size",
            "Records in the collector buffer",
        )
        self.attack_probability = Gauge(
            "ddos_attack_probability",
            "Latest attack probability from ensemble",
            ["model"],
        )
        self.collector_connected = Gauge(
            "ddos_collector_connected",
            "1 if collector is connected, 0 otherwise",
        )

        # --- Histograms ---
        self.tick_latency = Histogram(
            "ddos_tick_latency_seconds",
            "Time to process one tick",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )
        self.records_per_tick = Histogram(
            "ddos_records_per_tick",
            "Number of records processed per tick",
            buckets=(0, 10, 50, 100, 500, 1000, 5000, 10000),
        )
        self.confidence_distribution = Histogram(
            "ddos_confidence",
            "Confidence distribution of decisions",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
        )

        # --- Info ---
        self.build_info = Info(
            "ddos_defense",
            "Static metadata about the DDoS defense system",
        )
        self.build_info.info({
            "version": "1.0.0",
            "component": "ddos-defense",
        })

    # ------------------------------------------------------------------
    # Observation API
    # ------------------------------------------------------------------
    def observe_decision(self, ip: str, action: str, confidence: float,
                         source: str = "per_ip") -> None:
        self.decisions_total.labels(action=action, source=source).inc()
        self.confidence_distribution.observe(float(confidence))

    def observe_enforced_block(self, strike: int, model: str = "ensemble") -> None:
        if strike <= 1:
            bucket = "1"
        elif strike <= 4:
            bucket = "2-4"
        else:
            bucket = "5+"
        self.blocks_total.labels(model=model, strike_range=bucket).inc()

    def observe_whitelist_skip(self) -> None:
        self.whitelist_skips_total.inc()

    def observe_error(self, component: str) -> None:
        self.errors_total.labels(component=component).inc()

    def observe_tick(self, records: int, latency_seconds: float) -> None:
        self.tick_latency.observe(float(latency_seconds))
        self.records_per_tick.observe(float(records))

    def set_active_blocks(self, n: int) -> None:
        self.active_blocks.set(n)

    def set_permanent_blocks(self, n: int) -> None:
        self.permanent_blocks.set(n)

    def set_buffer_size(self, n: int) -> None:
        self.buffer_size.set(n)

    def set_collector_connected(self, connected: bool) -> None:
        self.collector_connected.set(1 if connected else 0)

    def set_attack_probability(self, prob: float, model: str = "ensemble") -> None:
        self.attack_probability.labels(model=model).set(float(prob))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

metrics = DDoSMetrics()


# ---------------------------------------------------------------------------
# Async push helper
# ---------------------------------------------------------------------------

_push_thread: threading.Thread | None = None
_push_lock = threading.Lock()


def _push_once() -> None:
    """Send current registry state to Pushgateway."""
    try:
        push_to_gateway(PUSHGATEWAY_URL, job=PUSHGATEWAY_JOB, registry=REGISTRY)
    except Exception as e:
        log.debug(f"push_to_gateway failed: {e}")


def push_async() -> None:
    """Fire-and-forget push — spawns a daemon thread that returns immediately."""
    global _push_thread
    with _push_lock:
        # If a push is already in-flight, skip (avoid piling up)
        if _push_thread is not None and _push_thread.is_alive():
            return
        _push_thread = threading.Thread(target=_push_once, daemon=True,
                                        name="metrics-push")
        _push_thread.start()


# ---------------------------------------------------------------------------
# Convenience: attach to a Runner instance
# ---------------------------------------------------------------------------

def wire_runner(runner) -> None:
    """Hook the metrics singleton into a Runner's tick method.

    Call this after creating a Runner:
        from monitoring.metrics import wire_runner
        wire_runner(runner)
    """
    original_tick = runner.tick

    def instrumented_tick():
        result = original_tick()
        try:
            records = result.get("records", 0)
            elapsed_ms = result.get("elapsed_ms", 0)
            metrics.observe_tick(records, elapsed_ms / 1000.0)

            blocks = result.get("blocks", 0)
            for _ in range(blocks):
                metrics.observe_enforced_block(strike=1, model="ensemble")

            # Update gauges from live state
            try:
                counts = runner.manager.state.count()
                metrics.set_active_blocks(counts["active"])
                metrics.set_permanent_blocks(counts["permanent"])
            except Exception:
                pass

            try:
                stats = runner.collector.stats()
                metrics.set_buffer_size(stats["buffer_size"])
                metrics.set_collector_connected(True)
            except Exception:
                pass

        except Exception as e:
            log.warning(f"metrics wiring error: {e}")

        # Async push — does NOT block the tick
        push_async()
        return result

    runner.tick = instrumented_tick
    log.info("metrics wired to runner")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser(description="Metrics endpoint")
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    log.info(f"starting /metrics endpoint on {args.host}:{args.port}")
    start_http_server(args.port, addr=args.host)

    print(f"Metrics available at http://{args.host}:{args.port}/metrics")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nstopping")