# mlops-azure-template

Production-ready **MLOps pipeline** for training, evaluating, monitoring and deploying 
ML models on **Azure ML** — with automated CI/CD via GitHub Actions.

## Architecture

```
Data → Train → Evaluate → Quality Gate → Register → Deploy → Monitor
                                                         ↓
                                               Drift Detection → Retrain
```

## Features

- End-to-end ML pipeline — train, evaluate, deploy
- MLflow experiment tracking and model registry
- Statistical drift detection using KS test and PSI
- Quality gates — model only deploys if metrics pass thresholds
- Azure ML integration — compute, endpoints, model registry
- GitHub Actions CI/CD — test, build, deploy automatically
- Docker containerisation
- Full Pytest test suite

## Quickstart

```bash
git clone https://github.com/Anne07-Ai/mlops-azure-template.git
cd mlops-azure-template

cp .env.example .env
# Add your Azure credentials to .env

pip install -r requirements.txt

# Train the model
python -m model.train

# Evaluate the model
python -m model.evaluate

# Run drift detection
python -c "
from model.train import generate_sample_data
from monitoring.drift_detector import DriftDetector
detector = DriftDetector()
ref = generate_sample_data(500)
cur = generate_sample_data(500)
detector.set_reference(ref)
print(detector.detect_drift(cur))
"
```

## CI/CD pipeline

Every push to `main` automatically:
1. Runs full Pytest test suite
2. Trains the model
3. Evaluates against quality thresholds
4. Builds Docker container
5. Deploys to Azure ML endpoint

## Quality gates

| Metric | Minimum threshold |
|--------|------------------|
| Accuracy | 0.75 |
| F1 Score | 0.70 |
| ROC AUC | 0.80 |

## Drift detection

Uses two statistical methods per feature:
- **KS Test** — detects distribution shift (p-value < 0.05)
- **PSI** — Population Stability Index (threshold > 0.15)

## Run tests

```bash
pytest tests/ -v
```

## Run with Docker

```bash
docker build -t mlops-azure-template .
docker run mlops-azure-template
```

## Tech stack

- Python 3.11, Scikit-learn, MLflow
- Azure ML, Azure Identity
- Evidently AI, SciPy
- Docker, GitHub Actions, Pytest