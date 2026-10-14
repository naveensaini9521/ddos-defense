python3 -m ml.train --data data/raw --model xgb --version v1_xgb --labels data/labeled/labels.jsonl

python3 -m ml.train --data data/raw --model iso --version v1_iso --contamination 0.3
python3 -m ml.train --data data/raw --model rf --version v1_rf
python3 -m ml.train --data data/raw --model xgb --version v1_xgb

## Test 1 — Tune XGBoost (small search)

bash
cd /var/www/ddos_defense
source .venv/bin/activate

python3 -m ml.hyperparameter_tuner --model xgb --search random --n-iter 10 --cv-folds 2

## Test 2 — Tune with augmentation

bash
python3 -m ml.hyperparameter_tuner --model xgb --search random --n-iter 15 --augment --cv-folds 2

## Test 3 — Inspect the saved params

bash
cat ml/models/tuned_params.json

## Test 4 — Use tuned params in training

bash
python3 -c "
from ml.hyperparameter_tuner import load_tuned
tuned = load_tuned('xgb')
if tuned:
print('found tuned XGBoost params:')
for k, v in tuned['best_params'].items():
print(f' {k}: {v}')
else:
print('no tuned params found — run the tuner first')
"

# cross_validator

## Test 1 — Stratified 3-fold CV on XGBoost

bash
cd /var/www/ddos_defense
source .venv/bin/activate

python3 -m ml.cross_validator --model xgb --method stratified --folds 3

## Test 2 Compare methods

python3 -m ml.cross_validator --model xgb --method loo

## Test 3 — With augmentation + bootstrap

bash
python3 -m ml.cross_validator --model xgb --folds 3 --augment --bootstrap

🎯 Quick Reference Table
Task Command
Train XGBoost python3 -m ml.train --model xgb --version v1_xgb
Train RandomForest python3 -m ml.train --model rf --version v1_rf
Train IsolationForest python3 -m ml.train --model iso --version v1_iso --contamination 0.3
Tune hyperparameters python3 -m ml.hyperparameter_tuner --model xgb --n-iter 20
Cross-validate python3 -m ml.cross_validator --model xgb --method stratified --folds 5
Optimize threshold python3 -m ml.threshold_optimizer --model xgb --objective f1
Explain predictions python3 -m ml.explainability --model ml/models/v1_xgb.pkl
Detect drift python3 -m ml.drift_detector --current data/raw
Add feedback python3 -m ml.online_learning status
Retrain from feedback python3 -m ml.online_learning retrain --model xgb
Train LSTM python3 -m ml.sequence_model --train --model lstm
Graph botnet detection python3 -m ml.graph_detector --data data/raw
📚 Full Command Reference (By Purpose)
🟦 1. TRAINING — Build Models
Train XGBoost (main model)
bash
cd /var/www/ddos_defense
source .venv/bin/activate

python3 -m ml.train --data data/raw --model xgb --version v1_xgb
Train RandomForest
bash
python3 -m ml.train --data data/raw --model rf --version v1_rf
Train IsolationForest
bash
python3 -m ml.train --data data/raw --model iso --version v1_iso --contamination 0.3
Train all three at once
bash
python3 -m ml.train --data data/raw --model iso --version v1_iso --contamination 0.3
python3 -m ml.train --data data/raw --model rf --version v1_rf
python3 -m ml.train --data data/raw --model xgb --version v1_xgb
Train with custom labels file
bash
python3 -m ml.train --data data/raw --labels data/labeled/labels.jsonl --model xgb --version v2_xgb
Train with custom test size
bash
python3 -m ml.train --data data/raw --model xgb --version v1_xgb --test-size 0.2
🟩 2. HYPERPARAMETER TUNING
Random search (fast, default)
bash
python3 -m ml.hyperparameter_tuner --model xgb --search random --n-iter 20
Grid search (thorough, slow)
bash
python3 -m ml.hyperparameter_tuner --model xgb --search grid
Tune with augmentation
bash
python3 -m ml.hyperparameter_tuner --model xgb --search random --n-iter 20 --augment
Tune with fewer CV folds (faster)
bash
python3 -m ml.hyperparameter_tuner --model xgb --n-iter 10 --cv-folds 2
Inspect saved tuned params
bash
cat ml/models/tuned_params.json | python3 -m json.tool
🟨 3. CROSS-VALIDATION
Stratified K-fold (standard)
bash
python3 -m ml.cross_validator --model xgb --method stratified --folds 5
Leave-One-Out (small datasets)
bash
python3 -m ml.cross_validator --model xgb --method loo
Time-series split (temporal data)
bash
python3 -m ml.cross_validator --model xgb --method time-series --folds 5
With augmentation + bootstrap CI
bash
python3 -m ml.cross_validator --model xgb --folds 3 --augment --bootstrap
Ignore tuned params (baseline)
bash
python3 -m ml.cross_validator --model xgb --folds 3 --no-tuned
🟥 4. THRESHOLD OPTIMIZATION
Maximize F1 (balanced)
bash
python3 -m ml.threshold_optimizer --model xgb --objective f1
Maximize precision (few false positives)
bash
python3 -m ml.threshold_optimizer --model xgb --objective precision
Maximize recall (catch all attacks)
bash
python3 -m ml.threshold_optimizer --model xgb --objective recall
Minimize cost (custom)
bash
python3 -m ml.threshold_optimizer --model xgb --objective cost --fp-cost 10 --fn-cost 1
Youden's J (ROC-based)
bash
python3 -m ml.threshold_optimizer --model xgb --objective youden
With augmentation
bash
python3 -m ml.threshold_optimizer --model xgb --objective f1 --augment
Inspect saved thresholds
bash
cat ml/models/thresholds.json | python3 -m json.tool
🟪 5. EXPLAINABILITY (SHAP)
Explain all IPs in training data
bash
python3 -m ml.explainability --model ml/models/v1_xgb.pkl --data data/raw --top-k 3
Explain a single IP
bash
python3 -c "
from collector.nginx_parser import parse_file
from collector.aggregator import aggregate
from ml.explainability import explain_prediction
import joblib

model = joblib.load('ml/models/v1_xgb.pkl')
feats = aggregate(parse_file('data/raw/training.log'))

ex = explain_prediction(model, feats['10.0.0.100'], top_k=5)
print(ex.summary)
"
Global feature importance
bash
python3 -c "
import joblib
from ml.explainability import print_global_importance
print_global_importance(joblib.load('ml/models/v1_xgb.pkl'))
"
🟫 6. DRIFT DETECTION
Save current data as reference
bash
python3 -m ml.drift_detector --current data/raw --save-reference
Detect drift against saved reference
bash
python3 -m ml.drift_detector --current data/raw
Detect drift with custom thresholds
bash
python3 -m ml.drift_detector --current data/raw --ks-alpha 0.05 --psi-threshold 0.2
Inspect drift state + history
bash
cat ml/models/drift_state.json | python3 -m json.tool
cat ml/models/drift_reports.jsonl
🟦 7. ONLINE LEARNING (Feedback)
Check feedback status
bash
python3 -m ml.online_learning status
Add a feedback sample (programmatic)
bash
python3 -c "
from collector.nginx_parser import parse_file
from collector.aggregator import aggregate
from ml.online_learning import FeedbackStore

store = FeedbackStore()
feats = aggregate(parse*file('data/raw/training.log'))
store.add_correction('10.0.0.100', feats['10.0.0.100'], true_label=1, note='confirmed attack')
"
Retrain from feedback
bash
python3 -m ml.online_learning retrain --model xgb
Clear feedback store
bash
python3 -m ml.online_learning clear
Inspect feedback file
bash
cat data/labeled/feedback.jsonl
🟩 8. SEQUENCE MODEL (LSTM/GRU)
Install PyTorch (once)
bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
Train LSTM
bash
python3 -m ml.sequence_model --train --model lstm --seq-len 8 --epochs 30 --version seq_v1
Train GRU
bash
python3 -m ml.sequence_model --train --model gru --seq-len 8 --epochs 30 --version seq_gru_v1
Predict from saved sequence model
bash
python3 -c "
from ml.sequence_model import SequenceDetector
det = SequenceDetector('ml/models/seq_v1.pt')
print('model loaded:', det.cfg.model)
"
🟥 9. GRAPH BOTNET DETECTION
Install NetworkX (once)
bash
pip install networkx
Analyze logs for botnets
bash
python3 -m ml.graph_detector --data data/raw
Stricter clustering (all 3 attributes must match)
bash
python3 -m ml.graph_detector --data data/raw --min-shared 3
Looser clustering
bash
python3 -m ml.graph_detector --data data/raw --min-shared 1 --score-threshold 0.3
Save report
bash
python3 -m ml.graph_detector --data data/raw --save
Inspect report
bash
cat ml/models/graph_report.json | python3 -m json.tool
🟨 10. DATA INSPECTION
Load training data (quick check)
bash
python3 -m ml.data_loader --raw data/raw --labels data/labeled/labels.jsonl
Check class balance
bash
python3 -c "
from ml.data_loader import load_training_data
from ml.augmenter import class_balance
X, y, * = load_training_data()
print(class_balance(y))
"
Test augmentation
bash
python3 -m ml.augmenter
🟪 11. INFERENCE TESTING
Test single-model detector
bash
python3 -c "
from collector.nginx_parser import parse_file
from collector.aggregator import aggregate
from ml.detector import Detector

det = Detector('ml/models/v1_xgb.pkl', threshold=0.5)
feats = aggregate(parse_file('data/raw/training.log'))

for ip, f in feats.items():
d = det.predict(f)
print(f'{ip:15s} -> {d.action:8s} conf={d.confidence:.2f}')
"
Test ensemble
bash
python3 -c "
from collector.nginx_parser import parse_file
from collector.aggregator import aggregate
from ml.ensemble import EnsembleDetector

det = EnsembleDetector(threshold=0.5)
feats = aggregate(parse_file('data/raw/training.log'))

for ip, f in feats.items():
d = det.predict(f)
print(f'{ip:15s} -> {d.action:8s} conf={d.confidence:.2f} {d.reason}')
"
Compare all three models
bash
python3 -c "
from collector.nginx_parser import parse_file
from collector.aggregator import aggregate
from ml.ensemble import EnsembleDetector

det = EnsembleDetector(threshold=0.5)
feats = aggregate(parse_file('data/raw/training.log'))

print(f'{\"IP\":15s} {\"iso\":>6s} {\"rf\":>6s} {\"xgb\":>6s} {\"ENSEMBLE\":>10s}')
print('-' \* 50)
for ip, f in feats.items():
iso = det.detectors['v1_iso'].predict(f).confidence
rf = det.detectors['v1_rf'].predict(f).confidence
xgb = det.detectors['v1_xgb'].predict(f).confidence
d = det.predict(f)
print(f'{ip:15s} {iso:6.2f} {rf:6.2f} {xgb:6.2f} {d.confidence:10.2f}')
"
🔄 Complete Training Workflow (Copy-Paste)
The full sequence from scratch:

bash
cd /var/www/ddos_defense
source .venv/bin/activate

# 1. Inspect data

python3 -m ml.data_loader --raw data/raw

# 2. Train all three models

python3 -m ml.train --data data/raw --model iso --version v1_iso --contamination 0.3
python3 -m ml.train --data data/raw --model rf --version v1_rf
python3 -m ml.train --data data/raw --model xgb --version v1_xgb

# 3. Tune XGBoost hyperparameters

python3 -m ml.hyperparameter_tuner --model xgb --search random --n-iter 20

# 4. Retrain with tuned params (uses tuned_params.json automatically)

python3 -m ml.train --data data/raw --model xgb --version v1_xgb

# 5. Cross-validate

python3 -m ml.cross_validator --model xgb --method stratified --folds 3

# 6. Optimize threshold

python3 -m ml.threshold_optimizer --model xgb --objective f1

# 7. Save drift reference

python3 -m ml.drift_detector --current data/raw --save-reference

# 8. Verify: ensemble predictions

python3 -c "
from collector.nginx_parser import parse_file
from collector.aggregator import aggregate
from ml.ensemble import EnsembleDetector
det = EnsembleDetector(threshold=0.5)
feats = aggregate(parse_file('data/raw/training.log'))
for ip, f in feats.items():
d = det.predict(f)
print(f'{ip:15s} -> {d.action:8s} conf={d.confidence:.2f}')
"

## analyze tarining log

python3 -m ml.global_features --file data/raw/training.log

## analyze botnet log

python3 -m ml.global_features --file data/raw/botnet_test.log

# Build-dataset

python3 -m ml.build_dataset --source data/collected --out data/labeled/features.csv

## Build the Dataset and Train

# 1. Build the training dataset from ALL collected files

python3 -m ml.build_dataset --source data/collected --out data/labeled/features.csv

# 2. Verify

wc -l data/labeled/features.csv
head -3 data/labeled/features.csv

# 3. Train v2 models on real data

python3 -m ml.train --data data/labeled/features.csv --model xgb --version v2_xgb
python3 -m ml.train --data data/labeled/features.csv --model rf --version v2_rf
python3 -m ml.train --data data/labeled/features.csv --model iso --version v2_iso --contamination 0.2

# 4. Compare v1 vs v2

python3 -c "
import joblib
from ml.data*loader import load_features_from_csv
X, y, * = load_features_from_csv('data/labeled/features.csv')
print(f'{int(y.sum())} attack / {int((y==0).sum())} normal')
print()
for v in ['v1_xgb', 'v2_xgb', 'v2_rf', 'v2_iso']:
try:
m = joblib.load(f'ml/models/{v}.pkl')
p = m.predict(X)
p_bin = (p == -1).astype(int) if 'iso' in v else (p == 1).astype(int)
acc = (p_bin == y).mean()
caught = int(((p_bin == 1) & (y == 1)).sum())
total_atk = int((y == 1).sum())
print(f'{v:10s} accuracy={acc:.3f} caught={caught}/{total_atk}')
except FileNotFoundError:
print(f'{v:10s} NOT TRAINED')
"
