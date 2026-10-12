"""Combine HTTP-log features with packet-capture features.

Produces a 42-feature vector per IP:
    - 12 HTTP features (from nginx logs)
    - 30 packet features (from pcap)

Usage:
    from ml.combined_features import extract_combined

    vec = extract_combined(http_features, packet_features)
"""
from __future__ import annotations

from dataclasses import asdict

from collector.aggregator import IPFeatures
from collector.packet_features import PACKET_FEATURE_NAMES, PacketFeatures
from core.schema import FEATURE_NAMES as HTTP_FEATURE_NAMES


# Combined feature name order
COMBINED_FEATURE_NAMES = list(HTTP_FEATURE_NAMES) + list(PACKET_FEATURE_NAMES)


def extract_combined(
    http_feats: IPFeatures | None,
    packet_feats: PacketFeatures | None,
) -> list[float]:
    """Combine HTTP and packet features into one vector.

    Missing features default to 0.0.
    """
    vec: list[float] = []

    # HTTP features (12)
    if http_feats:
        vec.extend(http_feats.to_vector())
    else:
        vec.extend([0.0] * len(HTTP_FEATURE_NAMES))

    # Packet features (30)
    if packet_feats:
        vec.extend(packet_feats.to_vector())
    else:
        vec.extend([0.0] * len(PACKET_FEATURE_NAMES))

    return vec


def combined_feature_names() -> list[str]:
    return list(COMBINED_FEATURE_NAMES)