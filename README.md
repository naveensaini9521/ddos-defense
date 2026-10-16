# DDoS Defense System

An end-to-end, ML-powered DDoS detection and mitigation system. Runs on a
controller host, tails logs from a remote VM, uses an ensemble of ML models
to detect attacks in real time, and enforces iptables rules on the VM.

Handles both **single-IP floods** (per-IP ML) and **rotating-IP botnets**
(ASN/subnet detection via ipset).

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [VM Setup (Victim)](#vm-setup-victim)
5. [Controller Setup](#controller-setup)
6. [Running the System](#running-the-system)
7. [Testing Dynamic IP Blocking](#testing-dynamic-ip-blocking)
8. [Docker Deployment](#docker-deployment)
9. [Monitoring](#monitoring)
10. [Attack Scenarios](#attack-scenarios)
11. [Configuration Reference](#configuration-reference)
12. [Troubleshooting](#troubleshooting)

---

## What It Does

Given a **victim VM** (target) and a **controller host**:

1. **Collects logs** — tails nginx access logs from the VM via SSH (with PTY
   for correct line buffering)
2. **Enriches each IP** — subnet (/24), ASN (via pyasn), ASN type
   (datacenter/mobile/residential)
3. **Extracts features** — 12 HTTP features per IP per window
4. **Detects attacks** — ensemble of 3 ML models (Iso + RF + XGB) for per-IP
5. **Detects botnets** — subnet concentration, ASN concentration,
   datacenter-share, new-IP rate
6. **Enforces blocks** — iptables on the VM (single IP) or ipset
   (entire ASN in one rule)
7. **Auto-unblocks** — Redis native TTL + exponential backoff
8. **Exposes metrics** — Prometheus endpoint + Grafana dashboards + alerts
9. **Provides API** — FastAPI control plane for queries and manual actions

### Key Features

- **Ensemble ML** — 3 models vote (Iso + RF + XGB) for per-IP decisions
- **Dynamic IP blocking** — blocks an entire ASN with ONE ipset rule
  (e.g. 4,401 prefixes for M247/NordVPN in 0.7s)
- **Safety layers** — whitelist + never-block ASNs (Google, Cloudflare,
  Akamai, AWS, DigitalOcean) + prefix caps + private-subnet guard
- **Persistent** — Redis (hot path) + SQLite (cold path)
- **Explainable** — SHAP values per decision
- **Self-improving** — online learning from analyst feedback
- **Drift-aware** — detects when the model needs retraining
- **Tested** — 64 tests, all passing

---

## Architecture

```text
┌───────────────────────────────────────────────────────────────┐
│ CONTROLLER HOST (192.168.122.1)                               │
│                                                               │
│ simulator/ ──attack──▶ VM                                     │
│                                                               │
│ pipeline/live_collector ──ssh -tt tail──▶ VM nginx logs       │
│        │                                                      │
│        ▼                                                      │
│ collector/aggregator (12 features)                            │
│        │                                                      │
│ pipeline/dynamic_ip (subnet + ASN concentration)              │
│        │                                                      │
│ ml/ensemble (ISO + RF + XGB)                                  │
│        │                                                      │
│ blocker/block_manager ──┬─▶ Redis (state)                     │
│                         └─▶ Firewall ──ssh──▶ VM iptables     │
│                                       └──▶ VM ipset           │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│ VICTIM VM (192.168.122.30)                                    │
│ nginx :80 ──▶ victim_server :8080                             │
│ logs → /var/log/nginx/ddos_access.log                         │
│ iptables INPUT chain (DROP rules)                             │
│ ipset ddos_asN (bulk prefix sets)                             │
└───────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone <repo> ddos-defense && cd ddos-defense

# 2. Install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp config/config.example.yaml config/config.yaml
# edit: remote_host, log_path, redis

# 4. Build ASN database (optional but recommended)
#    see ASN Setup section below

# 5. Run the defense loop
python3 -m pipeline.runner

# 6. (in another terminal) fire an attack
python3 -m simulator.attack_scenarios \
    --scenario dynamic_ip \
    --target http://192.168.122.30/ \
    --duration 30 --threads 30
```

---

## VM Setup (Victim)

Run these on the victim VM (or use `deploy/install.sh`).

### 1. Install nginx + Python

```bash
sudo apt-get update
sudo apt-get install -y nginx python3 python3-venv
```

### 2. Create the victim HTTP server

```bash
cat > ~/victim_server.py <<'EOF'
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time, json, sys

class H(BaseHTTPRequestHandler):
    def _h(self):
        body = json.dumps({"ok": True, "ts": time.time()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    do_GET = do_POST = _h
    def log_message(self, *a): pass

ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
EOF

nohup python3 ~/victim_server.py > ~/victim.log 2>&1 &
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/
# → 200
```

### 3. Configure nginx with XFF trust + log format

```bash
sudo tee /etc/nginx/sites-available/ddos-lab > /dev/null <<'EOF'
log_format ddos '$remote_addr - $remote_user [$time_local] '
                '"$request" $status $body_bytes_sent '
                '"$http_referer" "$http_user_agent"';

server {
    listen 80 default_server;
    server_name _;

    set_real_ip_from 192.168.122.0/24;
    set_real_ip_from 127.0.0.1;
    real_ip_header X-Forwarded-For;
    real_ip_recursive on;

    access_log /var/log/nginx/ddos_access.log ddos;
    error_log  /var/log/nginx/ddos_error.log warn;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
EOF

sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/ddos-lab /etc/nginx/sites-enabled/ddos-lab
sudo nginx -t && sudo systemctl reload nginx
```

Verify:

```bash
curl -s -H 'X-Forwarded-For: 203.0.113.42' http://localhost/ > /dev/null
sleep 0.5
sudo tail -1 /var/log/nginx/ddos_access.log
# → 203.0.113.42 - - [...] "GET / HTTP/1.1" 200 ...
```

### 4. Passwordless sudo for iptables + ipset

```bash
sudo tee /etc/sudoers.d/ddos-defense > /dev/null <<EOF
$(whoami) ALL=(ALL) NOPASSWD: /usr/sbin/iptables, /usr/sbin/ip6tables, \
                               /usr/sbin/ipset, /sbin/iptables, /sbin/ipset
EOF
sudo chmod 0440 /etc/sudoers.d/ddos-defense
sudo visudo -c
```

Verify:

```bash
sudo -n iptables -L INPUT -n | head -2
sudo -n ipset list 2>&1 | head -2
```

### 5. Install ipset (for ASN blocking)

```bash
sudo apt-get install -y ipset
```

### 6. Install tcpdump controller (optional, for pcap capture)

```bash
sudo cp deploy/tcpdump_control.sh /usr/local/bin/tcpdump_control
sudo chmod +x /usr/local/bin/tcpdump_control
```

---

## Controller Setup

Run these on the controller host (192.168.122.1).

### 1. Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git redis-server \
                        iptables ipset curl
```

### 2. Clone + venv

```bash
cd /var/www
git clone <repo> ddos_defense
cd ddos_defense
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure SSH to VM (passwordless)

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 2>/dev/null
ssh-copy-id naveen@192.168.122.30

# Test
ssh -o BatchMode=yes naveen@192.168.122.30 "echo SSH_OK"
# → SSH_OK
```

### 4. Set up ASN database (for dynamic IP blocking)

```bash
# Download pyasn + database
pip install pyasn

cd /tmp
curl -sSL -o pyasn_dl.py https://raw.githubusercontent.com/hadiasghari/pyasn/master/pyasn-utils/pyasn_util_download.py
curl -sSL -o pyasn_conv.py https://raw.githubusercontent.com/hadiasghari/pyasn/master/pyasn-utils/pyasn_util_convert.py

python3 pyasn_dl.py --latestv4
python3 pyasn_conv.py --single $(ls rib.*.bz2 | head -1) data/ipasn.dat
rm -f rib.*.bz2 pyasn_dl.py pyasn_conv.py
```

Verify:

```bash
cd /var/www/ddos_defense
python3 -c "from collector.enrich import stats; print(stats())"
# → 'pyasn_loaded': True
```

### 5. Train models (if not already in `ml/models/`)

```bash
# Collect training data first (see "Collecting labeled training data" below)

# Build dataset
python3 -m ml.build_dataset --source data/collected \
    --out data/labeled/features.csv

# Trim to 12 HTTP features (matches inference)
python3 - <<'EOF'
import csv
from pathlib import Path
HTTP = ['pkt_rate','byte_rate','syn_ratio','avg_pkt_size','unique_dports',
        'conn_duration','http_reqs','error_ratio','src_ip_entropy',
        'ua_entropy','path_concentration','time_regularity']
cols = ['window_index','scenario','ip'] + HTTP + ['label']
rows = list(csv.DictReader(open('data/labeled/features.csv')))
with open('data/labeled/features_http12.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
EOF

# Train
python3 -m ml.train --data data/labeled/features_http12.csv --model iso --version v11_iso --contamination 0.2
python3 -m ml.train --data data/labeled/features_http12.csv --model rf  --version v11_rf
python3 -m ml.train --data data/labeled/features_http12.csv --model xgb --version v11_xgb

# Verify (should say 12 features each)
python3 -c "
import joblib
for v in ['v11_iso','v11_rf','v11_xgb']:
    m = joblib.load(f'ml/models/{v}.pkl')
    print(f'{v}: {m.n_features_in_} features')
"
```

### 6. Start Redis (if not already running)

```bash
sudo systemctl start redis-server
redis-cli ping
# → PONG
```

### 7. Configure the controller

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml`:

```yaml
redis:
  host: localhost
  port: 6379

collector:
  log_path: /var/log/nginx/ddos_access.log
  remote_log: true
  remote_host: naveen@192.168.122.30
  use_sudo: false

ml:
  ensemble:
    iso_path: ml/models/v11_iso.pkl
    rf_path: ml/models/v11_rf.pkl
    xgb_path: ml/models/v11_xgb.pkl
  weights:
    iso: 0.20
    rf: 0.35
    xgb: 0.45

blocker:
  backend: iptables
  dry_run: false
  remote_host: naveen@192.168.122.30
  use_ipset: true
  max_asn_prefixes: 5000
  asn_db_path: data/ipasn.dat
  whitelist:
    - 127.0.0.1
    - 192.168.122.1
    - 192.168.122.30
    - 10.0.0.0/8
    - 172.16.0.0/12
    - 192.168.0.0/16
```

---

## Running the System

### Sanity check first

```bash
cd /var/www/ddos_defense
source .venv/bin/activate

# 1. Redis
python3 -m blocker.redis_state ping

# 2. VM reachable
ssh -o BatchMode=yes naveen@192.168.122.30 "echo OK"

# 3. Firewall reachable
ssh naveen@192.168.122.30 "sudo -n iptables -L INPUT -n | head -2"

# 4. Model feature count
python3 -c "import joblib; print(joblib.load('ml/models/v11_xgb.pkl').n_features_in_)"
# → 12

# 5. Live collector (should print stats after 5s)
python3 -c "
import time
from pipeline.live_collector import LiveCollector
lc = LiveCollector('ssh://naveen@192.168.122.30/var/log/nginx/ddos_access.log')
lc.start(); time.sleep(5); print(lc.stats()); lc.stop()
"
```

### Run the defense loop

```bash
python3 -m pipeline.runner --dry-run --duration 60
```

You'll see tick output every 10s:

```text
tick 1: no records in last 10s
tick 2: records=3940 ips=533 per_ip=0 subnet=1 blocks=1 (47675ms)
```

### Switch to live enforcement

Edit `config/config.yaml`:

```yaml
blocker:
  dry_run: false
```

Restart the runner.

---

## Testing Dynamic IP Blocking

This is the key test — a rotating-IP attack where no single IP looks bad,
but the aggregate pattern is clearly malicious.

### Step 1 — Clean slate

```bash
cd /var/www/ddos_defense
source .venv/bin/activate

# Reset state
python3 -m blocker.redis_state clear
rm -f data/firewall_blocked.txt data/firewall_asn_index.json

# Flush VM firewall
ssh naveen@192.168.122.30 "sudo -n iptables -F INPUT"
ssh naveen@192.168.122.30 "sudo -n ipset destroy 2>/dev/null || true"
```

### Step 2 — Terminal 1: run the runner

```bash
python3 -m pipeline.runner --duration 60
```

### Step 3 — Terminal 2: fire the attack (within 10s)

```bash
python3 -m simulator.attack_scenarios \
    --scenario dynamic_ip \
    --target http://192.168.122.30/ \
    --duration 30 --threads 30
```

### Step 4 — Watch Terminal 1

You should see:

```text
dynamic IP signal: 592 IPs, top ASN 9009 (90%), dc_share=100%
ASN BLOCK DECISION: AS9009 (datacenter) share=90% dc_share=100% ttl=300s
tick 4: subnet/ASN block active — skipping 592 per-IP decisions
AS9009 → 4401 IPv4 prefixes
BLOCKED AS9009 via ipset (4401 prefixes, 0.70s)
ENFORCED block on AS9009  scope=asn  ttl=5.0min
tick 4: records=5728 ips=592 per_ip=0 subnet=1 blocks=1 (724ms)
```

What just happened:

- 592 unique IPs, each sending 1-2 requests → per-IP ML couldn't flag them
- Detector saw 90% concentration on ASN 9009 (M247, NordVPN's ASN)
- Blocked the entire ASN — 4,401 prefixes in ONE ipset rule

### Step 5 — Verify on the VM

```bash
# ipset holds all 4,401 prefixes for AS9009
ssh naveen@192.168.122.30 "sudo -n ipset list ddos_as9009 | head -8"

# One iptables rule references the set
ssh naveen@192.168.122.30 "sudo -n iptables -L INPUT -n"
# DROP  all  --  0.0.0.0/0  0.0.0.0/0  match-set ddos_as9009 src
```

### Step 6 — Cleanup

```bash
ssh naveen@192.168.122.30 "sudo -n ipset destroy ddos_as9009"
ssh naveen@192.168.122.30 "sudo -n iptables -F INPUT"
python3 -m blocker.redis_state clear
```

### Comparing attack modes

| Scenario                   | Source IPs | Detected by          | Blocked by            |
| -------------------------- | ---------- | -------------------- | --------------------- |
| `http_flood`               | 1          | Per-IP ML            | Single iptables rule  |
| `dynamic_ip` (90% pool)    | 592+       | ASN concentration    | ipset (4401 prefixes) |
| `dynamic_ip` (equal pools) | 592+       | Datacenter share     | Top datacenter ASN    |
| `distributed`              | fake XFF   | Subnet concentration | /24                   |

---

## Docker Deployment

The entire stack runs in Docker Compose: Redis, runner, API, metrics
exporter, Prometheus, Pushgateway, Grafana.

### Prerequisites

- Docker + Docker Compose v2
- Docker Desktop file sharing must include `/var/www`
  (Settings → Resources → File Sharing)

### Start

```bash
cd /var/www/ddos_defense
docker compose up -d --build
docker compose ps
```

### Load Prometheus + Grafana configs (one-time, after first start)

Docker named volumes are used, so configs must be copied in:

```bash
# Prometheus
docker cp monitoring/prometheus/prometheus.yml ddos-prometheus:/etc/prometheus/
docker cp monitoring/prometheus/alerts.yml     ddos-prometheus:/etc/prometheus/
docker exec ddos-prometheus kill -HUP 1

# Grafana dashboards + datasource
docker exec ddos-grafana mkdir -p /etc/grafana/provisioning/datasources
docker exec ddos-grafana mkdir -p /etc/grafana/provisioning/dashboards
docker cp monitoring/grafana/provisioning/datasources/prometheus.yml \
    ddos-grafana:/etc/grafana/provisioning/datasources/
docker cp monitoring/grafana/provisioning/dashboards/default.yml \
    ddos-grafana:/etc/grafana/provisioning/dashboards/
docker cp monitoring/grafana/provisioning/dashboards/ddos.json \
    ddos-grafana:/etc/grafana/provisioning/dashboards/
docker restart ddos-grafana
```

### Verify

```bash
docker compose ps
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool | grep health
curl -s http://localhost:9100/metrics | grep -c '^ddos_'
```

### Access

| Service     | URL                   | Credentials   |
| ----------- | --------------------- | ------------- |
| Grafana     | http://localhost:3001 | admin / admin |
| Prometheus  | http://localhost:9090 | —             |
| API         | http://localhost:8000 | —             |
| Pushgateway | http://localhost:9091 | —             |

### SSH to VM from runner container

The runner container needs `~/.ssh` mounted. This is already configured in
`docker-compose.yml`. The container reaches the VM via the host's
`192.168.122.x` network — no special config needed if the host has the route.

If not reachable:

```yaml
runner:
  network_mode: host
```

### Stop / clean

```bash
docker compose down           # stop, keep volumes
docker compose down -v        # stop + wipe volumes
```

---

## Monitoring

### Metrics

Prometheus scrapes 3 targets:

- `ddos-api:8000` — API process metrics
- `ddos-metrics:9100` — custom exporter (Redis + firewall state)
- `pushgateway:9091` — runner metrics (pushed each tick)

Key metrics:

| Metric                       | Type      | Meaning                        |
| ---------------------------- | --------- | ------------------------------ |
| `ddos_decisions_total`       | counter   | Decisions by action + source   |
| `ddos_blocks_total`          | counter   | Blocks by model + strike range |
| `ddos_active_blocks`         | gauge     | Current active block count     |
| `ddos_tick_latency_seconds`  | histogram | Per-tick processing time       |
| `ddos_attack_probability`    | gauge     | Latest ensemble P(attack)      |
| `ddos_collector_buffer_size` | gauge     | Records in collector buffer    |

### Alerts (in `monitoring/alerts.yml`)

| Alert                   | Trigger                    | Severity |
| ----------------------- | -------------------------- | -------- |
| `HighBlockRate`         | > 10 blocks/s for 30s      | warning  |
| `VeryHighBlockRate`     | > 50 blocks/s for 15s      | critical |
| `ManyActiveBlocks`      | > 100 concurrent           | warning  |
| `CollectorDisconnected` | buffer empty 30s           | critical |
| `HighPipelineLatency`   | P95 tick > 1s              | warning  |
| `HighConfidenceAttacks` | avg P(attack) > 0.9 for 2m | critical |

### Grafana

The dashboard (auto-provisioned) shows:

- Block Rate (blocks/sec)
- Active Blocks
- Tick Latency (P50/P95/P99)
- Blocks/sec over time
- Decisions/sec by action
- Collector health

### Alerts (production)

For alerting to Slack/email, add Alertmanager:

```yaml
# docker-compose.yml
alertmanager:
  image: prom/alertmanager:v0.27.0
  ports: ["9093:9093"]
  volumes:
    - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
```

---

## Attack Scenarios

All scenarios live in `simulator/attack_scenarios.py`.

| Scenario        | Description                                          |
| --------------- | ---------------------------------------------------- |
| `http_flood`    | Single-IP volumetric flood (tests per-IP ML)         |
| `slowloris`     | Connection exhaustion via slow reads                 |
| `ramping_flood` | Gradually increasing rate (tests temporal detection) |
| `bursty_flood`  | On/off burst pattern                                 |
| `distributed`   | XFF-spoofed requests (fake IPs)                      |
| `mixed`         | Normal traffic + attack blend                        |
| `dynamic_ip`    | Rotating IPs, weighted 90% to one ASN                |

### Running any scenario

```bash
python3 -m simulator.attack_scenarios \
    --scenario dynamic_ip \
    --target http://192.168.122.30/ \
    --duration 30 \
    --threads 30
```

### Collecting labeled training data

```bash
python3 -m simulator.collector \
    --scenario dynamic_ip \
    --target http://192.168.122.30/ \
    --vm-host naveen@192.168.122.30 \
    --duration 15 --warmup 3 --cooldown 3
```

Output: `data/collected/<timestamp>_dynamic_ip.{log,jsonl,meta.json,pcap}`

---

## Configuration Reference

### `config/config.yaml` — key fields

```yaml
redis:
  host: localhost
  port: 6379

collector:
  log_path: /var/log/nginx/ddos_access.log
  remote_log: true
  remote_host: naveen@192.168.122.30
  use_sudo: false
  window_seconds: 10
  max_age: 3600

ml:
  ensemble:
    iso_path: ml/models/v11_iso.pkl
    rf_path: ml/models/v11_rf.pkl
    xgb_path: ml/models/v11_xgb.pkl
  weights:
    iso: 0.20
    rf: 0.35
    xgb: 0.45
  threshold: 0.7

blocker:
  backend: iptables
  dry_run: false # true = log-only, false = real enforcement
  remote_host: naveen@192.168.122.30
  use_ipset: true # true = fast ASN blocks via ipset
  max_asn_prefixes: 5000 # refuse to block larger ASNs
  asn_db_path: data/ipasn.dat
  whitelist: [...] # never-block list

pipeline:
  window_seconds: 10
  dynamic_ip:
    enabled: true
    subnet_size: 24
    new_ip_threshold: 20
```

### `dry_run` semantics

| Value   | Behavior                                     |
| ------- | -------------------------------------------- |
| `true`  | Decisions logged, iptables/ipset NOT touched |
| `false` | Real enforcement on the VM                   |

Use `true` for testing, `false` for production.

---

## Troubleshooting

### Runner shows `no records in last 10s`

Check nginx logs on the VM:

```bash
ssh naveen@192.168.122.30 "sudo -n tail -5 /var/log/nginx/ddos_access.log"
```

Check the collector:

```bash
python3 -c "import time; from pipeline.live_collector import LiveCollector; lc=LiveCollector('ssh://naveen@192.168.122.30/var/log/nginx/ddos_access.log'); lc.start(); time.sleep(5); print(lc.stats()); lc.stop()"
```

If `lines=0` → SSH PTY issue. Verify this works:

```bash
ssh -tt naveen@192.168.122.30 "tail -1 /var/log/nginx/ddos_access.log"
```

### `ensemble weights must sum to a positive value`

`config.ml.weights` keys must match `model_paths` keys (`iso`, `rf`, `xgb`),
**not** `v1_iso`, `v1_rf`, `v1_xgb`.

### `X has 12 features, but IsolationForest is expecting 9`

Retrain models on the current schema (12 features):

```bash
python3 -m ml.train --data data/labeled/features_http12.csv --model xgb --version v11_xgb
```

### `redis-py not installed`

```bash
pip install redis
```

Or in Docker: add `redis` to `requirements.txt` and rebuild.

### `iptables: Bad rule (does a matching rule exist in that chain?)`

Harmless — cosmetic. `unblock()` treats this as success.

### Docker: `address already in use`

Host services occupy ports. Kill them:

```bash
sudo pkill prometheus
sudo pkill grafana-server
pkill -f monitoring.exporters.custom_metrics
sudo systemctl stop redis-server
```

### Docker: `not a directory` on config mount

Docker Desktop file-sharing issue. Use named volumes (see
`docker-compose.yml`) or move the project under `$HOME`.

### Grafana shows "No data"

Confirm Prometheus has the data:

```bash
curl -s 'http://localhost:9090/api/v1/query?query=ddos_blocks_total'
```

If empty, Pushgateway isn't being scraped. Check:

```bash
curl -s http://localhost:9091/metrics | grep ddos_
curl -s http://localhost:9090/api/v1/targets | grep -A2 pushgateway
```

---

## Project Structure

```text
ddos_defense/
├── api/                    FastAPI control plane
│   ├── app.py
│   ├── schemas.py
│   └── routes/
│       ├── health.py
│       ├── blocks.py
│       ├── stats.py
│       └── admin.py
├── blocker/                Enforcement
│   ├── block_manager.py    Orchestrator
│   ├── firewall.py         iptables + ipset
│   ├── redis_state.py      Persistent state
│   ├── state.py            SQLite fallback
│   ├── expiry.py           TTL + backoff
│   ├── whitelist.py
│   └── subnet_blocker.py
├── collector/              Log parsing + features
│   ├── nginx_parser.py
│   ├── aggregator.py       12 HTTP features
│   ├── enrich.py           Subnet + ASN
│   └── packet_features.py  30 pcap features (offline)
├── ml/                     Machine learning
│   ├── ensemble.py         3-model voting
│   ├── detector.py
│   ├── train.py
│   ├── build_dataset.py
│   └── models/             Trained .pkl files
├── pipeline/               Runtime
│   ├── runner.py           Main loop
│   ├── live_collector.py   SSH PTY tail
│   └── dynamic_ip.py       ASN/subnet detector
├── simulator/              Attack generation
│   ├── attack_scenarios.py
│   └── collector.py        Labeled data collection
├── monitoring/             Observability
│   ├── prometheus/
│   ├── exporters/
│   ├── grafana/
│   └── metrics.py
├── deploy/                 Deployment helpers
│   ├── install.sh
│   ├── remote_blocker.py
│   └── tcpdump_control.sh
├── config/
│   └── config.yaml
├── data/                   Runtime state
│   ├── collected/          Training data
│   ├── labeled/
│   └── ipasn.dat           ASN database
├── tests/                  64 tests
├── docs/                   Architecture + diagrams
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## License

Naveen Saini
