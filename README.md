# DDoS Defense System

An end-to-end, ML-powered DDoS detection and mitigation system. Runs on a
controller host, tails logs from a remote VM, uses an ensemble of ML models
to detect attacks in real time, and enforces iptables rules on the VM.

Built for learning how real-world DDoS defense works — from raw logs to
subnet-level blocking.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Running](#running)
6. [Components](#components)
7. [Attack Types Detected](#attack-types-detected)
8. [Configuration](#configuration)
9. [Testing](#testing)
10. [Monitoring](#monitoring)
11. [Troubleshooting](#troubleshooting)
12. [Project Structure](#project-structure)

---

## What It Does

Given a target server (VM) and a controller host:

1. **Collects logs** — tails nginx access logs from the VM via SSH
2. **Extracts features** — turns raw log lines into per-IP feature vectors
3. **Detects attacks** — using an ensemble of 3 ML models (IsolationForest, RandomForest, XGBoost)
4. **Detects botnets** — using global features + subnet analysis
5. **Enforces blocks** — applies iptables DROP rules on the VM via SSH
6. **Auto-unblocks** — TTL + exponential backoff for repeat offenders
7. **Exposes metrics** — Prometheus endpoint + alert rules
8. **Provides API** — HTTP control plane for querying and manual actions

### Key Features

- **Ensemble ML** — 3 models vote (Iso + RF + XGB) for high accuracy
- **Handles dynamic IPs** — detects and blocks rotating botnets by subnet
- **Safety first** — whitelist prevents locking yourself out
- **Persistent** — survives restarts (SQLite)
- **Explainable** — SHAP values for every decision
- **Self-improving** — online learning from analyst feedback
- **Drift-aware** — detects when the model needs retraining
- **Tested** — 48 tests covering every module

---

## Architecture

```text
┌───────────────────────────────────────────────────────────────┐
│                          VICTIM VM                            │
│                                                               │
│  nginx ──▶ /var/log/nginx/access.log                          │
│                                                               │
│  victim_server.py (port 8080)                                 │
│                                                               │
│  iptables rules applied remotely by controller                │
└─────────────────────┬─────────────────────────────────────────┘
                      │ SSH tail -F
                      ▼
┌───────────────────────────────────────────────────────────────┐
│                       CONTROLLER HOST                         │
│                                                               │
│  LiveCollector                                                │
│      ↓                                                        │
│  Aggregator ──▶ Ensemble (Iso+RF+XGB) ──▶ per-IP decisions    │
│      ↓                                                        │
│  GlobalFeatures ──▶ SubnetBlocker ──▶ subnet decisions        │
│                                                               │
│  Both decision streams ──▶ BlockManager                       │
│                              ├─▶ Whitelist check              │
│                              ├─▶ State DB                     │
│                              ├─▶ Expiry (TTL/backoff)         │
│                              └─▶ iptables (via SSH)           │
│                                                               │
│  FastAPI control plane  (port 8000)                           │
│  Prometheus /metrics    (port 9090)                           │
└───────────────────────────────────────────────────────────────┘
```

---

## Quick Start

**Assumes:** Linux host with two network-reachable machines — a controller
(your laptop) and a target VM.

```bash
# 1. Clone / copy project
cd /var/www/ddos_defense

# 2. Setup Python env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# 3. Set up the victim VM (once)
scp deploy/install.sh naveen@192.168.122.30:~/
ssh -t naveen@192.168.122.30 "bash ~/install.sh"

# 4. Configure
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml — set remote_host, whitelist

# 5. Train models (once, or when you have new data)
python3 -m ml.train --data data/raw --model iso --version v1_iso --contamination 0.3
python3 -m ml.train --data data/raw --model rf  --version v1_rf
python3 -m ml.train --data data/raw --model xgb --version v1_xgb

# 6. Run the pipeline (dry-run first)
python3 -m pipeline.runner --dry-run --duration 60

# 7. Enable real blocking (edit config: blocker.dry_run = false)
python3 -m pipeline.runner
```

---

## Installation

### Prerequisites

**On the host (controller):**

- Linux (Ubuntu 22.04+ recommended)
- Python 3.10+
- SSH key-based access to the VM

**On the VM (victim):**

- Linux (Ubuntu 22.04+)
- sudo access
- Port 8080 open for the backend

### Host Setup

```bash
cd /var/www/ddos_defense

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### VM Setup

```bash
# From the host
scp deploy/install.sh naveen@<VM_IP>:~/

# Run interactively (you'll be prompted for sudo password once)
ssh -t naveen@<VM_IP> "bash ~/install.sh"
```

The install script:

- Installs packages (nginx, python3, curl)
- Sets up passwordless sudo for iptables
- Configures nginx to proxy to port 8080
- Deploys `victim_server.py`
- Fixes log file permissions
- Runs end-to-end verification

### Verify Setup

```bash
# From the host

# 1. SSH works passwordless
ssh naveen@<VM_IP> "echo SSH_OK"

# 2. Passwordless iptables works
ssh naveen@<VM_IP> "sudo -n iptables -L INPUT -n | head -2"

# 3. nginx → victim_server works
curl -s http://<VM_IP>/ | head -c 100

# 4. Log file gets traffic
curl -s http://<VM_IP>/ > /dev/null
ssh naveen@<VM_IP> "tail -1 /var/log/nginx/access.log"
```

---

## Running

### The Main Loop

```bash
python3 -m pipeline.runner
```

The runner will:

1. Start the log collector (SSH tail)
2. Load the 3 ML models
3. Initialize the blocker
4. Loop forever, one tick every 10 seconds

### Modes

| Command | Behavior |
|---|---|
| `python3 -m pipeline.runner` | Run forever |
| `python3 -m pipeline.runner --once` | Run one tick and exit |
| `python3 -m pipeline.runner --dry-run` | Log decisions, don't enforce |
| `python3 -m pipeline.runner --duration 60` | Run for 60 seconds |

### Running Components Separately

**Collector only (no blocking):**

```bash
python3 -m pipeline.live_collector \
    --source ssh://naveen@<VM_IP>/var/log/nginx/access.log \
    --duration 30
```

**Detector only:**

```bash
python3 -c "
from collector.nginx_parser import parse_file
from collector.aggregator import aggregate
from ml.ensemble import EnsembleDetector

det = EnsembleDetector(threshold=0.7)
feats = aggregate(parse_file('data/raw/training.log'))
for ip, f in feats.items():
    d = det.predict(f)
    print(f'{ip:15s} → {d.action:8s} (conf={d.confidence:.2f})')
"
```

**API server:**

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Metrics endpoint:**

```bash
python3 -m monitoring.metrics --port 9090
```

---

## Components

### 1. Collector

Tails logs and parses them into structured records.

| Module | Purpose |
|---|---|
| `collector/nginx_parser.py` | Parse nginx log lines (supports multiple time formats) |
| `collector/aggregator.py` | Group records by IP, compute 9-dim feature vectors |
| `pipeline/live_collector.py` | SSH tail with auto-reconnect |
| `deploy/remote_collector.py` | Robust SSH tail with preflight checks |

### 2. Machine Learning

| Module | Purpose |
|---|---|
| `ml/train.py` | Train Iso / RF / XGB models |
| `ml/detector.py` | Single-model inference wrapper |
| `ml/ensemble.py` | 3-model weighted-vote ensemble |
| `ml/feature_engineering.py` | Temporal features (10 extra per IP) |
| `ml/data_loader.py` | Load labeled training data |
| `ml/augmenter.py` | SMOTE-style synthetic attack generation |
| `ml/hyperparameter_tuner.py` | Random/Grid search for best params |
| `ml/cross_validator.py` | K-fold CV with bootstrap CI |
| `ml/threshold_optimizer.py` | Find optimal blocking threshold |
| `ml/explainability.py` | SHAP values for decisions |
| `ml/drift_detector.py` | Detect model staleness (KS + PSI) |
| `ml/online_learning.py` | Feedback-driven retraining |
| `ml/sequence_model.py` | LSTM for temporal patterns |
| `ml/graph_detector.py` | Botnet clustering via graph analysis |
| `ml/global_features.py` | Cross-IP features (rotation, churn, entropy) |

### 3. Blocker

| Module | Purpose |
|---|---|
| `blocker/whitelist.py` | Never-block list (safety) |
| `blocker/state.py` | SQLite persistence of blocks |
| `blocker/expiry.py` | TTL + exponential backoff |
| `blocker/firewall.py` | iptables wrapper (local or SSH) |
| `blocker/block_manager.py` | Orchestrator |
| `blocker/subnet_blocker.py` | Block whole /24s for botnet attacks |

### 4. Pipeline

| Module | Purpose |
|---|---|
| `pipeline/live_collector.py` | SSH tail → parsed records |
| `pipeline/runner.py` | Main defense loop |
| `pipeline/dynamic_ip.py` | Detect rotating IP attacks |

### 5. API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `GET /blocks/` | List active blocks |
| `GET /blocks/{ip}` | Get specific block |
| `POST /blocks/block` | Manually block an IP |
| `POST /blocks/{ip}/unblock` | Manually unblock |
| `GET /stats/summary` | Stats summary |
| `GET /stats/history` | Block/unblock history |
| `POST /admin/reload-model` | Trigger model reload |
| `POST /admin/purge-expired` | Clean expired records |

### 6. Deployment

| Module | Purpose |
|---|---|
| `deploy/remote_collector.py` | Preflight + robust SSH tail |
| `deploy/remote_blocker.py` | Preflight + robust iptables |
| `deploy/install.sh` | One-shot VM setup |

### 7. Monitoring

| Module | Purpose |
|---|---|
| `monitoring/metrics.py` | Prometheus metrics |
| `monitoring/alerts.yml` | Alert rules (11 total) |

---

## Attack Types Detected

| Attack | Detection Layer | Action |
|---|---|---|
| HTTP flood | Ensemble (per-IP) | Block IP |
| Slowloris | Ensemble + IsolationForest | Block IP |
| SYN flood | Per-IP features | Block IP |
| Rotating botnet | GlobalFeatures + SubnetBlocker | Block /24 |
| Coordinated subnet attack | SubnetBlocker | Block /24 |
| Distributed low-rate | GlobalFeatures | Block subnet |
| Unknown zero-day | IsolationForest | Block IP |
| Spoofed UA/path | Entropy analysis | Block subnet |

---

## Configuration

All config lives in `config/config.yaml`:

```yaml
app:
  name: ddos-defense
  log_level: INFO

collector:
  window_seconds: 10
  source: nginx
  log_path: /var/log/nginx/access.log
  remote_log: true
  remote_host: naveen@192.168.122.30
  use_sudo: false

ml:
  model_path: ml/models/v1_xgb.pkl
  ensemble:
    iso_path: ml/models/v1_iso.pkl
    rf_path: ml/models/v1_rf.pkl
    xgb_path: ml/models/v1_xgb.pkl
  threshold: 0.7
  weights:
    v1_iso: 0.20
    v1_rf: 0.35
    v1_xgb: 0.45

blocker:
  backend: iptables
  chain: INPUT
  rule_target: DROP
  dry_run: true                      # ← flip to false for real blocking
  remote_host: naveen@192.168.122.30
  remote_sudo: true

  default_ttl: 60
  backoff_seconds:
    - 60
    - 300
    - 1800
    - 86400
  permanent_after_strike: 5
  max_ttl: 86400

  whitelist:
    - 127.0.0.1
    - 192.168.122.30
    - 10.0.0.0/8
    - 172.16.0.0/12
    - 192.168.0.0/16
```

### Important Settings

| Setting | Meaning |
|---|---|
| `ml.threshold` | Block if P(attack) >= this (0.7 = balanced) |
| `blocker.dry_run` | `true` = log only, `false` = real iptables |
| `blocker.whitelist` | IPs/ranges never blocked |
| `blocker.backoff_seconds` | TTL per strike (escalates) |
| `blocker.permanent_after_strike` | When to stop escalating |
| `pipeline.dynamic_ip.enabled` | Enable subnet blocking |

---

## Testing

```bash
# Activate env
source .venv/bin/activate

# Run all tests
pytest

# Verbose
pytest -v

# With coverage
pytest --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_blocker.py -v

# Run a specific test
pytest tests/test_blocker.py::TestState::test_permanent_block -v
```

### Test Coverage

| Test file | Tests | Covers |
|---|---|---|
| `test_blocker.py` | 21 | Whitelist, state, expiry, firewall |
| `test_pipeline.py` | 14 | Parser, aggregator, collector, dynamic IP |
| `test_dynamic_ip.py` | 9 | Subnet blocking, global features |
| `test_collector.py` | 1 | Parser smoke test |
| `test_detector.py` | 1 | Detector interface |
| `test_features.py` | 1 | Feature schema |
| `test_integration.py` | 1 | End-to-end |
| **Total** | **48** | — |

All tests should pass in ~2 seconds.

---

## Monitoring

### Prometheus

```bash
# Start /metrics endpoint
python3 -m monitoring.metrics --port 9090
```

Then add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: ddos-defense
    static_configs:
      - targets: ['localhost:9090']
```

### Key Metrics

| Metric | Type | Meaning |
|---|---|---|
| `ddos_decisions_total` | Counter | Decisions by action/source |
| `ddos_blocks_total` | Counter | Blocks enforced |
| `ddos_active_blocks` | Gauge | Currently blocked IPs |
| `ddos_tick_latency_seconds` | Histogram | Processing time per tick |
| `ddos_attack_probability` | Gauge | Latest P(attack) |
| `ddos_collector_connected` | Gauge | 1 if connected, 0 if not |

### Alerts

11 alert rules in `monitoring/alerts.yml`:

- **HighBlockRate** — > 10 blocks/sec
- **VeryHighBlockRate** — > 50 blocks/sec (critical)
- **CollectorDisconnected** — pipeline is blind
- **HighPipelineLatency** — processing too slow
- **PipelineErrors** — errors/sec > 0
- ... (see file for full list)

---

## Troubleshooting

### "no records in last 10s"

**Cause:** Collector isn't reading logs.

**Check:**

```bash
# 1. VM reachable?
ping <VM_IP>

# 2. SSH works?
ssh naveen@<VM_IP> "echo OK"

# 3. Log file exists?
ssh naveen@<VM_IP> "ls -la /var/log/nginx/access.log"

# 4. Direct tail works?
ssh naveen@<VM_IP> "timeout 3 tail -F -n 0 /var/log/nginx/access.log" &
curl http://<VM_IP>/ > /dev/null
sleep 1
kill %1
```

### "sudo: interactive authentication is required"

**Cause:** Passwordless sudo not configured on the VM.

**Fix:** Re-run `install.sh`, or manually:

```bash
sudo tee /etc/sudoers.d/ddos-defense > /dev/null <<EOF
$(whoami) ALL=(ALL) NOPASSWD: /usr/sbin/iptables, /usr/sbin/ip6tables
EOF
sudo chmod 0440 /etc/sudoers.d/ddos-defense
sudo visudo -c
```

### "from __future__ imports must occur at the beginning of the file"

**Cause:** A comment or code above the docstring/code.

**Fix:** Ensure line 1 is the docstring and line 2+ is
`from __future__ import annotations`. Delete and re-create the file cleanly.

### Blocks not appearing in iptables

**Cause:** `blocker.dry_run: true` in config.

**Fix:** Set to `false`, restart the pipeline.

### Pipeline blocks my SSH session

**Cause:** Your IP isn't whitelisted.

**Fix:** Add it to `blocker.whitelist` in config. If you're already locked
out, SSH from a different machine and remove the rule manually:

```bash
ssh <other-machine>
ssh naveen@<VM_IP>
sudo iptables -D INPUT -s <your-ip> -j DROP
```

### Restarting from scratch

```bash
# Clean up
rm -f data/blocker_state.db
rm -f data/firewall_blocked.txt
ssh naveen@<VM_IP> "sudo iptables -F INPUT"    # careful!

# Reinstall VM
scp deploy/install.sh naveen@<VM_IP>:~/
ssh -t naveen@<VM_IP> "bash ~/install.sh"
```

---

## Project Structure

```text
ddos-defense/
├── README.md                    ← this file
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
│
├── config/
│   ├── config.yaml              ← main config
│   └── config.example.yaml
│
├── core/                        # shared primitives
│   ├── schema.py                # feature names, Decision, BlockRecord
│   ├── logging.py               # JSON logging
│   ├── config_loader.py
│   └── types.py
│
├── simulator/
│   ├── traffic_generator.py     # attack traffic generator
│   ├── scenarios.py
│   └── configs/                 # scenario YAMLs
│
├── collector/
│   ├── nginx_parser.py          # parse log lines
│   ├── aggregator.py            # per-IP features
│   └── fixtures/
│
├── ml/                          # 15 modules
│   ├── features.py
│   ├── train.py
│   ├── detector.py
│   ├── ensemble.py
│   ├── feature_engineering.py
│   ├── data_loader.py
│   ├── augmenter.py
│   ├── hyperparameter_tuner.py
│   ├── cross_validator.py
│   ├── threshold_optimizer.py
│   ├── explainability.py
│   ├── drift_detector.py
│   ├── online_learning.py
│   ├── sequence_model.py
│   ├── graph_detector.py
│   ├── global_features.py
│   └── models/                  # trained .pkl files
│
├── blocker/                     # 6 modules
│   ├── whitelist.py
│   ├── state.py
│   ├── expiry.py
│   ├── firewall.py
│   ├── block_manager.py
│   └── subnet_blocker.py
│
├── pipeline/                    # 3 modules
│   ├── live_collector.py
│   ├── runner.py
│   └── dynamic_ip.py
│
├── api/                         # 7 modules
│   ├── app.py
│   ├── schemas.py
│   └── routes/
│       ├── health.py
│       ├── blocks.py
│       ├── stats.py
│       └── admin.py
│
├── deploy/                      # 3 modules
│   ├── remote_collector.py
│   ├── remote_blocker.py
│   └── install.sh               # VM setup script
│
├── monitoring/                  # 3 modules
│   ├── metrics.py
│   ├── alerts.yml
│   └── grafana/
│
├── data/
│   ├── raw/                     # training logs
│   ├── labeled/                 # labels.jsonl
│   ├── processed/
│   └── blocker_state.db         # SQLite
│
└── tests/                       # 7 test files, 48 tests
    ├── test_blocker.py
    ├── test_pipeline.py
    ├── test_dynamic_ip.py
    ├── test_collector.py
    ├── test_detector.py
    ├── test_features.py
    └── test_integration.py
```

---

## License

Educational / learning project. Use freely for learning DDoS defense.

> ⚠️ **Warning:** Only run against systems you own. Unauthorized DDoS
> attacks are illegal in most jurisdictions and can result in criminal
> prosecution.

---

## Acknowledgments

Built as a complete learning exercise covering:

- **Networking** (SSH, TCP/IP, iptables)
- **Machine learning** (ensemble methods, anomaly detection, drift)
- **Systems** (SQLite, threads, subprocess)
- **Security** (whitelists, rate limiting, spoofing)
- **Ops** (systemd, Prometheus, deployment)
- **Testing** (pytest, coverage, fixtures)
