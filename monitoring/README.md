## Test 1 — Start the /metrics server

cd /var/www/ddos_defense
source .venv/bin/activate

pip install prometheus-client # if not already

python3 -m monitoring.metrics --port 9090

## Terminal 2 — fetch metrics:

curl -s http://localhost:9090/metrics | head -30

## python3 -c "

from monitoring.metrics import metrics

# Simulate some activity

metrics.observe_decision('1.2.3.4', 'block', 0.95, 'per_ip')
metrics.observe_decision('1.2.3.4', 'block', 0.95, 'per_ip')
metrics.observe_decision('5.6.7.8', 'allow', 0.10, 'per_ip')
metrics.observe_decision('203.0.113.0/24', 'block', 0.85, 'subnet')
metrics.observe_enforced_block(strike=1, model='ensemble')
metrics.observe_tick(50, 0.045)
metrics.set_active_blocks(3)
metrics.set_attack_probability(0.72, 'ensemble')

print('metrics updated')
"

### Test

# Terminal 1

python3 -m pipeline.runner --dry-run --duration 30 2>&1 | tee /tmp/runner.log

# Terminal 2 (within 10s)

python3 -m simulator.attack_scenarios --scenario dynamic_ip \
 --target http://192.168.122.30/ --duration 20 --threads 30
