"""Tests for collector parsers + enrichment."""
from __future__ import annotations

from collector.nginx_parser import parse_line, parse_file
from collector.enrich import enrich, subnet_of, asn_of


def test_nginx_parser():
    line = '1.2.3.4 - - [01/Oct/2026:11:13:10 +0000] "GET / HTTP/1.1" 200 615'
    r = parse_line(line)
    assert r is not None
    assert r.ip == "1.2.3.4"
    assert r.status == 200
    assert r.bytes == 615


def test_parser_handles_ua():
    line = ('1.2.3.4 - - [01/Oct/2026:11:13:10 +0000] "GET / HTTP/1.1" 200 100 '
            '"-" "curl/8.5.0"')
    r = parse_line(line)
    assert r.user_agent == "curl/8.5.0"


def test_subnet_of():
    assert subnet_of("203.0.113.42") == "203.0.113.0/24"
    assert subnet_of("8.8.8.8") == "8.8.8.0/24"


def test_enrich_test_ranges():
    """Curated ASN table maps the RFC 5737 test ranges."""
    assert asn_of("203.0.113.42") == 9009       # M247
    assert asn_of("198.51.100.7") == 16509      # AWS
    assert asn_of("192.0.2.99") == 14061        # DigitalOcean


def test_enrich_returns_all_keys():
    ctx = enrich("203.0.113.42")
    assert set(ctx.keys()) >= {"ip", "subnet", "asn", "asn_type", "asn_name", "ja3"}
    assert ctx["asn_type"] == "datacenter"


def test_parse_file_missing_returns_empty():
    assert parse_file("/tmp/does-not-exist-xyz.log") == []