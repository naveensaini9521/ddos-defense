"""Tests for blocker track: whitelist, state, expiry, firewall."""
from __future__ import annotations

import time
import tempfile
from pathlib import Path

import pytest

from blocker.expiry import ExpiryDaemon, ExpiryPolicy, PERMANENT, format_ttl
from blocker.firewall import Firewall, FirewallConfig
from blocker.state import BlockState
from blocker.whitelist import Whitelist


class TestWhitelist:
    def test_default_ips(self):
        wl = Whitelist()
        assert wl.is_whitelisted("127.0.0.1")
        assert wl.is_whitelisted("::1")

    def test_add_and_check(self):
        wl = Whitelist()
        wl.add_ip("1.2.3.4")
        assert wl.is_whitelisted("1.2.3.4")
        assert not wl.is_whitelisted("5.6.7.8")

    def test_add_network(self):
        wl = Whitelist()
        wl.add_network("10.0.0.0/24")
        assert wl.is_whitelisted("10.0.0.5")
        assert wl.is_whitelisted("10.0.0.200")
        assert not wl.is_whitelisted("10.0.1.5")

    def test_invalid_input_ignored(self):
        wl = Whitelist()
        wl.add_ip("not-an-ip")
        wl.add_network("not-a-cidr")
        assert not wl.is_whitelisted("not-an-ip")

    def test_filter_allowed(self):
        wl = Whitelist()
        wl.add_ip("10.0.0.1")
        candidates = ["10.0.0.1", "1.2.3.4", "5.6.7.8"]
        allowed = wl.filter_allowed(candidates)
        assert "10.0.0.1" not in allowed
        assert "1.2.3.4" in allowed

    def test_from_config(self):
        cfg = {"blocker": {"whitelist": ["8.8.8.8", "10.0.0.0/8"]}}
        wl = Whitelist.from_config(cfg)
        assert wl.is_whitelisted("8.8.8.8")
        assert wl.is_whitelisted("10.5.5.5")


class TestState:
    @pytest.fixture
    def state(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "test.db"
            s = BlockState(db)
            yield s
            s.close()

    def test_add_and_check(self, state):
        state.add_block("1.2.3.4", ttl=60, reason="test")
        assert state.is_blocked("1.2.3.4")
        assert not state.is_blocked("5.6.7.8")

    def test_expiry(self, state):
        state.add_block("1.2.3.4", ttl=0.1, reason="test")
        assert state.is_blocked("1.2.3.4")
        time.sleep(0.2)
        assert not state.is_blocked("1.2.3.4")

    def test_permanent_block(self, state):
        state.add_block("1.2.3.4", ttl=-1, reason="perm")
        entry = state.get("1.2.3.4")
        assert entry.is_permanent
        assert entry.is_active

    def test_strike_escalation(self, state):
        e1 = state.add_block("1.2.3.4", ttl=60, reason="a")
        e2 = state.add_block("1.2.3.4", ttl=60, reason="b")
        e3 = state.add_block("1.2.3.4", ttl=60, reason="c")
        assert e1.strike == 1
        assert e2.strike == 2
        assert e3.strike == 3

    def test_purge_keeps_permanent(self, state):
        state.add_block("1.2.3.4", ttl=0.1, reason="short")
        state.add_block("5.6.7.8", ttl=-1, reason="perm")
        time.sleep(0.2)
        n = state.purge_expired()
        assert n == 1
        assert state.get("5.6.7.8") is not None
        assert state.get("1.2.3.4") is None

    def test_active_blocks(self, state):
        state.add_block("1.1.1.1", ttl=60)
        state.add_block("2.2.2.2", ttl=-1)
        active = state.active_blocks()
        ips = [e.ip for e in active]
        assert "1.1.1.1" in ips
        assert "2.2.2.2" in ips

    def test_remove(self, state):
        state.add_block("1.2.3.4", ttl=60)
        assert state.remove_block("1.2.3.4")
        assert not state.is_blocked("1.2.3.4")
        assert not state.remove_block("nonexistent")


class TestExpiryPolicy:
    def test_default_schedule(self):
        p = ExpiryPolicy()
        assert p.ttl_for(1) == 60
        assert p.ttl_for(2) == 300
        assert p.ttl_for(3) == 1800
        assert p.ttl_for(4) == 86400
        assert p.ttl_for(5) == PERMANENT
        assert p.ttl_for(10) == PERMANENT

    def test_custom_schedule(self):
        p = ExpiryPolicy(backoff=[30, 60], permanent_threshold=3)
        assert p.ttl_for(1) == 30
        assert p.ttl_for(2) == 60
        assert p.ttl_for(3) == PERMANENT

    def test_max_ttl_cap(self):
        p = ExpiryPolicy(backoff=[999999], max_ttl=100)
        assert p.ttl_for(1) == 100

    def test_format_ttl(self):
        assert format_ttl(30) == "30s"
        assert format_ttl(120) == "2.0min"
        assert "hr" in format_ttl(3600)
        assert "day" in format_ttl(86400)
        assert format_ttl(PERMANENT) == "permanent"


class TestExpiryDaemon:
    @pytest.fixture
    def state(self):
        with tempfile.TemporaryDirectory() as d:
            s = BlockState(Path(d) / "test.db")
            yield s
            s.close()

    def test_daemon_unblocks_expired(self, state):
        state.add_block("1.2.3.4", ttl=0.3)

        unblocked: list[str] = []
        daemon = ExpiryDaemon(
            state,
            on_unblock=lambda ip: unblocked.append(ip),
            check_interval=0.1,
        )
        daemon.start()
        try:
            time.sleep(1.0)
        finally:
            daemon.stop()

        assert "1.2.3.4" in unblocked
        assert not state.is_blocked("1.2.3.4")

    def test_tick_once(self, state):
        state.add_block("1.2.3.4", ttl=0.05)
        time.sleep(0.1)
        daemon = ExpiryDaemon(state, check_interval=999)
        n = daemon.tick_once()
        assert n >= 1


class TestFirewall:
    def test_dry_run_block(self, tmp_path, monkeypatch):
        import blocker.firewall as fw_mod
        monkeypatch.setattr(fw_mod, "BLOCKED_FILE", tmp_path / "blocked.txt")

        fw = Firewall(FirewallConfig(dry_run=True))
        r = fw.block("1.2.3.4")
        assert r.ok
        assert "1.2.3.4" in fw.list_blocked()

        r = fw.unblock("1.2.3.4")
        assert r.ok
        assert "1.2.3.4" not in fw.list_blocked()

    def test_double_block_is_idempotent(self, tmp_path, monkeypatch):
        import blocker.firewall as fw_mod
        monkeypatch.setattr(fw_mod, "BLOCKED_FILE", tmp_path / "blocked.txt")

        fw = Firewall(FirewallConfig(dry_run=True))
        fw.block("1.2.3.4")
        r2 = fw.block("1.2.3.4")
        assert r2.ok
        assert fw.count() == 1