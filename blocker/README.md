# Analyze botnet log

cd /var/www/ddos_defense
source .venv/bin/activate

find . -type d -name "**pycache**" -exec rm -rf {} + 2>/dev/null

python3 -m blocker.subnet_blocker --file data/raw/botnet_test.log

## Botnet from public-looking range

python3 -m blocker.subnet_blocker --file data/raw/botnet_public.log

## Normal log produces no block

python3 -m blocker.subnet_blocker --file data/raw/training.log
