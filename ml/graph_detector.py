"""Graph-based botnet detection.

Builds a graph where:
    - Nodes  = source IPs
    - Edges  = shared characteristics (User-Agent, paths, timing)

Botnets form dense clusters in this graph. We detect them by:
    - Finding connected components (basic clustering)
    - Computing clustering coefficient per community
    - Detecting communities (Louvain algorithm)
    - Scoring each cluster as "suspicious" based on features

Usage:
    python3 -m ml.graph_detector --data data/raw
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from collector.nginx_parser import ParsedLog, parse_file
from core.logging import get_logger

log = get_logger("ml.graph_detector")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    ips: list[str]
    size: int
    density: float                    # edges / possible edges
    shared_attributes: list[str]      # what made them cluster
    attack_score: float               # 0..1
    reason: str = ""


@dataclass
class GraphReport:
    n_nodes: int
    n_edges: int
    n_components: int
    clusters: list[Cluster] = field(default_factory=list)
    suspicious_ips: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Graph Botnet Report ===",
            f"nodes:      {self.n_nodes}",
            f"edges:      {self.n_edges}",
            f"components: {self.n_components}",
            f"clusters:   {len(self.clusters)}",
            f"suspicious: {len(self.suspicious_ips)} IPs",
            "",
        ]
        for c in self.clusters:
            lines.append(f"  cluster of {c.size} IPs  "
                         f"score={c.attack_score:.2f}  "
                         f"reason={c.reason}")
            for ip in c.ips[:5]:
                lines.append(f"    - {ip}")
            if c.size > 5:
                lines.append(f"    ... +{c.size - 5} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    records: list[ParsedLog],
    min_shared: int = 2,
) -> tuple[nx.Graph, dict]:
    """Build IP-cooccurrence graph.

    Two IPs get an edge if they share >= min_shared of:
        - User-Agent
        - Path
        - Timing bucket (same 5-second window)
    """
    G = nx.Graph()

    # Collect attributes per IP
    per_ip: dict[str, dict] = defaultdict(lambda: {
        "user_agents": set(),
        "paths": set(),
        "time_buckets": set(),
        "requests": 0,
    })

    for r in records:
        ip = r.ip
        d = per_ip[ip]
        d["user_agents"].add(r.user_agent or "unknown")
        d["paths"].add(r.path)
        d["time_buckets"].add(int(r.ts // 5))   # 5-second buckets
        d["requests"] += 1

    ips = list(per_ip.keys())

    # Add nodes with attributes
    for ip in ips:
        G.add_node(ip, **{
            "user_agents": per_ip[ip]["user_agents"],
            "paths": per_ip[ip]["paths"],
            "time_buckets": per_ip[ip]["time_buckets"],
            "requests": per_ip[ip]["requests"],
        })

    # Add edges based on shared attributes
    for i, ip_a in enumerate(ips):
        for ip_b in ips[i + 1:]:
            a, b = per_ip[ip_a], per_ip[ip_b]

            shared = 0
            shared_attrs = []

            if a["user_agents"] & b["user_agents"]:
                shared += 1
                shared_attrs.append("user_agent")
            if a["paths"] & b["paths"]:
                shared += 1
                shared_attrs.append("path")
            if a["time_buckets"] & b["time_buckets"]:
                shared += 1
                shared_attrs.append("timing")

            if shared >= min_shared:
                G.add_edge(ip_a, ip_b,
                           weight=shared,
                           shared=shared_attrs)

    log.info(f"built graph: {G.number_of_nodes()} nodes, "
             f"{G.number_of_edges()} edges")
    return G, per_ip


# ---------------------------------------------------------------------------
# Cluster analysis
# ---------------------------------------------------------------------------

def detect_clusters(
    G: nx.Graph,
    per_ip: dict,
    min_cluster_size: int = 2,
) -> list[Cluster]:
    """Find connected components (or communities) and score them."""
    clusters: list[Cluster] = []

    # Use connected components (simple, deterministic)
    components = list(nx.connected_components(G))

    for comp in components:
        if len(comp) < min_cluster_size:
            continue

        subgraph = G.subgraph(comp)
        density = nx.density(subgraph) if len(comp) > 1 else 0.0

        # Aggregate shared attributes across the cluster
        shared_attrs_counter: dict[str, int] = defaultdict(int)
        for _, _, data in subgraph.edges(data=True):
            for attr in data.get("shared", []):
                shared_attrs_counter[attr] += 1

        shared_attrs = sorted(shared_attrs_counter.keys(),
                              key=lambda k: -shared_attrs_counter[k])

        # Suspicion score
        score = _cluster_score(comp, subgraph, per_ip, shared_attrs)

        clusters.append(Cluster(
            ips=sorted(comp),
            size=len(comp),
            density=density,
            shared_attributes=shared_attrs,
            attack_score=score,
            reason=_describe_reason(shared_attrs, score),
        ))

    # Sort by score, highest first
    clusters.sort(key=lambda c: -c.attack_score)
    return clusters


def _cluster_score(
    nodes: set,
    subgraph: nx.Graph,
    per_ip: dict,
    shared_attrs: list[str],
) -> float:
    """Score a cluster as potential botnet (0..1)."""
    score = 0.0

    # 1. Cluster size (bigger = more suspicious, up to a point)
    size = len(nodes)
    if size >= 5:
        score += 0.30
    elif size >= 3:
        score += 0.15

    # 2. Density (fully-connected = coordinated)
    if subgraph.number_of_nodes() > 1:
        score += 0.20 * nx.density(subgraph)

    # 3. Diversity penalty — botnets hit few paths, use few User-Agents
    all_paths = set()
    all_uas = set()
    for ip in nodes:
        all_paths |= per_ip[ip]["paths"]
        all_uas |= per_ip[ip]["user_agents"]

    # If cluster has very few distinct paths relative to size → suspicious
    paths_per_ip = len(all_paths) / max(size, 1)
    if paths_per_ip < 1.5:
        score += 0.20
    elif paths_per_ip < 3:
        score += 0.10

    # If cluster uses very few UAs → suspicious
    uas_per_ip = len(all_uas) / max(size, 1)
    if uas_per_ip < 1.2:
        score += 0.20
    elif uas_per_ip < 2:
        score += 0.10

    # 4. Bonus for timing coordination
    if "timing" in shared_attrs:
        score += 0.10

    return min(score, 1.0)


def _describe_reason(shared_attrs: list[str], score: float) -> str:
    if score < 0.3:
        return f"weak clustering (shared: {','.join(shared_attrs) or 'none'})"
    if score < 0.6:
        return f"moderate clustering (shared: {','.join(shared_attrs)})"
    return f"strong botnet signature (shared: {','.join(shared_attrs)})"


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def analyze(records: list[ParsedLog],
            min_shared: int = 2,
            min_cluster_size: int = 2,
            score_threshold: float = 0.5) -> GraphReport:
    """Run the full graph analysis pipeline."""
    if not records:
        return GraphReport(0, 0, 0)

    G, per_ip = build_graph(records, min_shared=min_shared)

    # Basic stats
    n_components = nx.number_connected_components(G)
    clusters = detect_clusters(G, per_ip, min_cluster_size=min_cluster_size)

    # Suspicious IPs from high-scoring clusters
    suspicious: list[str] = []
    for c in clusters:
        if c.attack_score >= score_threshold:
            suspicious.extend(c.ips)

    return GraphReport(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        n_components=n_components,
        clusters=clusters,
        suspicious_ips=sorted(set(suspicious)),
    )


def analyze_logs(log_dir: str | Path = "data/raw",
                 **kwargs) -> GraphReport:
    """Convenience: analyze every .log file in a directory."""
    log_dir = Path(log_dir)
    all_recs: list[ParsedLog] = []
    for lf in sorted(log_dir.glob("*.log")):
        all_recs.extend(parse_file(lf))
    log.info(f"loaded {len(all_recs)} records from {log_dir}")
    return analyze(all_recs, **kwargs)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_report(report: GraphReport, path: str | Path = "ml/models/graph_report.json") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump({
            "timestamp": time.time(),
            "n_nodes": report.n_nodes,
            "n_edges": report.n_edges,
            "n_components": report.n_components,
            "clusters": [
                {
                    "ips": c.ips,
                    "size": c.size,
                    "density": c.density,
                    "shared_attributes": c.shared_attributes,
                    "attack_score": c.attack_score,
                    "reason": c.reason,
                }
                for c in report.clusters
            ],
            "suspicious_ips": report.suspicious_ips,
        }, f, indent=2)
    log.info(f"saved graph report to {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph botnet detection")
    p.add_argument("--data", default="data/raw")
    p.add_argument("--min-shared", type=int, default=2)
    p.add_argument("--min-cluster", type=int, default=2)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    report = analyze_logs(
        args.data,
        min_shared=args.min_shared,
        min_cluster_size=args.min_cluster,
        score_threshold=args.score_threshold,
    )

    print()
    print(report.summary())

    if args.save:
        save_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())