"""Tests for FastAPI endpoints."""
from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("fastapi not installed", allow_module_level=True)

from api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded", "down")
    assert "components" in data


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "ddos-defense"


def test_stats_summary(client):
    r = client.get("/stats/summary")
    assert r.status_code == 200
    data = r.json()
    assert "active_blocks" in data
    assert "state_backend" in data


def test_blocks_list(client):
    r = client.get("/blocks/")
    assert r.status_code == 200
    assert "blocks" in r.json()


def test_block_not_found(client):
    r = client.get("/blocks/255.255.255.254")
    assert r.status_code == 404


def test_blocks_asns(client):
    r = client.get("/stats/asns")
    assert r.status_code == 200
    assert "asns" in r.json()


def test_admin_state(client):
    r = client.get("/admin/state")
    assert r.status_code == 200
    assert "backend" in r.json()