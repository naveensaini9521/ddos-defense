"""Expose custom Prometheus metrics from collector + blocker.

This module provides a standalone HTTP exporter that serves /metrics
by pulling live state from the pipeline components.

Two ways to use:
    1. Programmatically:
         from monitoring.exporters.custom_metrics import serve
         serve(port=9100)

    2. As a script:
         python3 -m monitoring.exporters.custom_metrics --port 9100

The HTTP endpoint reads from:
    - RedisBlockState    → ddos_active_blocks, ddos_permanent_blocks
    - data/firewall_*.txt → ddos_blocked_ips_total, ddos_asn_blocks_total
    - LiveCollector      → ddos_collector_buffer_size (if wired)
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from core.config_loader import load as load_config
from core.logging import get_logger
from monitoring.metrics import metrics

log = get_logger("monitoring.exporters.custom_metrics")


def _refresh_from_state() -> None:
    """Update the metrics gauges from live state."""
    cfg = load_config()

    # --- Redis state ---
    try:
        redis_cfg = cfg.get("redis", {})
        if redis_cfg.get("host"):
            from blocker.redis_state import RedisBlockState
            state = RedisBlockState(
                host=redis_cfg["host"],
                port=int(redis_cfg.get("port", 6379)),
                db=int(redis_cfg.get("db", 0)),
                prefix=redis_cfg.get("prefix", "ddos:"),
            )
            counts = state.count()
            metrics.set_active_blocks(counts["active"])
            metrics.set_permanent_blocks(counts["permanent"])
            state.close()
        else:
            from blocker.state import BlockState
            state = BlockState()
            counts = state.count()
            metrics.set_active_blocks(counts["active"])
            metrics.set_permanent_blocks(counts["permanent"])
            state.close()
    except Exception as e:
        log.debug(f"state refresh failed: {e}")

    # --- Firewall files ---
    try:
        blocked_file = Path("data/firewall_blocked.txt")
        if blocked_file.exists():
            n = sum(
                1 for line in blocked_file.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            )
            metrics.set_buffer_size(n)  # reuse — or define a new gauge
    except Exception as e:
        log.debug(f"firewall refresh failed: {e}")


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        _refresh_from_state()
        body = generate_latest()

        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass   # silence default HTTP access log


def serve(host: str = "0.0.0.0", port: int = 9100) -> None:
    """Start the /metrics HTTP server (blocking)."""
    log.info(f"custom metrics server on {host}:{port}")
    srv = ThreadingHTTPServer((host, port), MetricsHandler)
    srv.serve_forever()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Custom metrics exporter")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9100)
    args = p.parse_args()

    print(f"serving /metrics on http://{args.host}:{args.port}/metrics")
    serve(args.host, args.port)