# ML-Forge: Design Document

**Project:** Production-grade, dataset-agnostic ML pipeline
**First dataset:** Titanic (Kaggle)
**Timeline:** Month 1 capstone → extended through Months 2-3
**Author:** Jason
**Status:** Planning

---

## 1. Vision

Most Titanic portfolio projects are single Kaggle notebooks. ML-Forge is a **full production ML system** — modular, tested, config-driven, and deployable. The Titanic dataset is the first dish; the kitchen handles any recipe.

**What this demonstrates to reviewers:**
- Software engineering discipline (modular code, tests, Docker, CI/CD)
- ML fundamentals (feature pipelines, proper evaluation, experiment tracking)
- Production thinking (API serving, config-driven design, monitoring scaffolding)

---

## 2. Architecture Overview

```
ml-forge/
├── configs/                  # Dataset-specific YAML configs
│   └── titanic.yaml
├── ml_forge/                 # Core Python package
│   ├── __init__.py
│   ├── cli.py                # Typer CLI entrypoint
│   ├── config.py             # Config loader + validation (Pydantic)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # Dataset loading + train/val/test splits
│   │   └── validator.py      # Great Expectations data validation (light)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base transformer class
│   │   ├── engineering.py    # Custom sklearn-compatible transformers
│   │   └── selection.py      # Feature selection strategies
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training orchestration
│   │   ├── evaluation.py     # Metrics, confusion matrix, classification report
│   │   ├── tuning.py         # Optuna hyperparameter optimization
│   │   └── stacking.py       # Ensemble/stacking meta-learner
│   ├── serving/
│   │   ├── __init__.py
│   │   └── api.py            # FastAPI prediction endpoint
│   └── utils/
│       ├── __init__.py
│       ├── logging.py        # Structured logging (structlog)
│       └── artifacts.py      # Model saving/loading with metadata
├── tracking/                 # MLflow experiment tracking (light)
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_models.py
│   │   └── test_config.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── conftest.py           # Shared fixtures
├── artifacts/                # Saved models + metadata (gitignored)
├── notebooks/                # Exploratory only (not the main deliverable)
│   └── eda.ipynb
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
├── .github/
│   └── workflows/
│       └── ci.yaml           # Lint + test + train on push
└── README.md                 # Project narrative + results
```

---

## 3. Depth Tiers

### Tier 1 — Deep Dive (Month 1 Core)

#### 3.1 Feature Engineering
**Goal:** Show you can think beyond `df.fillna()` and build reusable, testable transformers.

**Custom sklearn-compatible transformers:**
- `TitleExtractor` — parse titles (Mr, Mrs, Master, etc.) from Name, map to ordinal categories
- `FamilySizeBuilder` — combine SibSp + Parch into family size, create `IsAlone` binary flag
- `CabinDeckExtractor` — pull deck letter from Cabin, handle missing as separate category
- `TicketPrefixParser` — extract ticket prefix patterns (correlate with class/fare)
- `AgeBinner` — bin ages into meaningful groups (child/teen/adult/senior) using domain knowledge
- `FareBinner` — quantile-based fare bucketing

**Abstract base class (`base.py`):**
```python
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class BaseFeatureTransformer(ABC, BaseEstimator, TransformerMixin):
    """All custom transformers inherit from this."""

    @abstractmethod
    def fit(self, X, y=None):
        ...

    @abstractmethod
    def transform(self, X):
        ...

    def get_feature_names_out(self, input_features=None):
        """Required for sklearn pipeline introspection."""
        ...
```

**Feature selection:**
- Mutual information scores
- Recursive feature elimination (RFE)
- Permutation importance (post-training)
- Compare selected features across methods → document in README

**Design decisions to understand deeply:**
- Why sklearn's transformer API looks the way it does (fit/transform separation = preventing data leakage)
- Why `get_feature_names_out` matters for pipeline transparency
- When to use `ColumnTransformer` vs. custom transformers vs. both

---

#### 3.2 Modeling Depth
**Goal:** Go beyond "I tried Random Forest and got 82%" — show systematic experimentation.

**Model progression (each logged via MLflow):**
1. **Baseline:** Logistic Regression (interpretable benchmark)
2. **Tree-based:** Random Forest, Gradient Boosting (XGBoost, LightGBM)
3. **SVM:** with RBF kernel (contrast with linear models)
4. **Stacking ensemble:** meta-learner combining top performers

**Hyperparameter optimization (Optuna):**
- Define search spaces per model in YAML config
- Use `TPESampler` (explain why: Bayesian > random > grid)
- Pruning with `MedianPruner` for early stopping of bad trials
- Visualize optimization history (parallel coordinate plots, importance)

**Stacking architecture:**
```
Layer 0 (base learners):  LogReg | RF | XGBoost | LightGBM | SVM
                              ↓       ↓      ↓         ↓       ↓
                         [out-of-fold predictions via cross-validation]
                              ↓
Layer 1 (meta-learner):   Logistic Regression on stacked predictions
                              ↓
                         Final prediction
```

**Evaluation rigor:**
- Stratified k-fold CV (not just a single train/test split)
- Metrics: accuracy, precision, recall, F1, AUC-ROC, log loss
- Confusion matrix analysis — where does the model fail and why?
- Statistical significance: paired t-test across folds between models
- Learning curves to diagnose bias/variance

**Design decisions to understand deeply:**
- Why cross-validated stacking prevents leakage (out-of-fold predictions)
- Why TPE outperforms grid/random search (surrogate modeling analogy)
- How to read learning curves (is more data or more complexity the answer?)

---

#### 3.3 Software Engineering
**Goal:** Code that looks like it came from an engineering team, not a Jupyter notebook.

**Typer CLI (`cli.py`):**
```bash
ml-forge train --config configs/titanic.yaml --experiment baseline
ml-forge predict --model artifacts/best_model.pkl --input data/test.csv
ml-forge evaluate --model artifacts/best_model.pkl --config configs/titanic.yaml
ml-forge tune --config configs/titanic.yaml --n-trials 100
ml-forge serve --model artifacts/best_model.pkl --port 8000
```

**Config-driven design (Pydantic + YAML):**
```yaml
# configs/titanic.yaml
dataset:
  name: titanic
  train_path: data/train.csv
  test_path: data/test.csv
  target: Survived
  features:
    numeric: [Age, Fare, SibSp, Parch]
    categorical: [Pclass, Sex, Embarked]
    engineered: [FamilySize, IsAlone, Title, CabinDeck]

splitting:
  strategy: stratified_kfold
  n_splits: 5
  test_size: 0.2
  random_state: 42

models:
  logistic_regression:
    enabled: true
    params:
      C: 1.0
      max_iter: 1000
  random_forest:
    enabled: true
    params:
      n_estimators: 100
      max_depth: 10
  xgboost:
    enabled: true
    params:
      n_estimators: 200
      learning_rate: 0.1
      max_depth: 6

tuning:
  n_trials: 100
  timeout: 600
  sampler: tpe
  pruner: median
```

**Design patterns:**
- **Strategy pattern** for model selection (swap models via config, not code changes)
- **Factory pattern** for transformer/model instantiation
- **Abstract base classes** for transformers and evaluators

**Structured logging (structlog):**
- JSON-formatted logs for machine readability
- Log every pipeline stage: data loaded → features built → model trained → metrics
- Contextual info: timestamp, dataset name, model type, fold number

**Design decisions to understand deeply:**
- Why config-driven > hardcoded (reproducibility, collaboration, experiment tracking)
- When abstract base classes help vs. add unnecessary complexity
- Why structured logging matters in production (grepping JSON vs. parsing print statements)

---

### Tier 2 — Light/Functional (Month 1, deepen in Month 2+)

#### 3.4 Testing (Light)
**Month 1 scope:** Get the scaffolding right with meaningful tests, not exhaustive coverage.

- **pytest basics:** fixtures, parametrize, conftest
- **Unit tests:**
  - Transformers produce expected output shapes
  - Config validation catches bad YAML
  - Model training doesn't crash on toy data
- **One integration test:** full pipeline from raw CSV → prediction
- **Great Expectations (light):** basic data validation suite
  - Column exists, no unexpected nulls in critical columns, Fare >= 0, Age in range

**Month 2+ deepening:**
- Property-based testing with Hypothesis (fuzz transformer inputs)
- Coverage targets (>80%)
- Contract tests for the API layer
- Data validation as a pipeline gate (fail-fast on bad data)

---

#### 3.5 MLOps (Light)
**Month 1 scope:** Working infrastructure, not production-hardened.

- **MLflow tracking:**
  - Log params, metrics, and model artifacts per experiment
  - Local tracking server (no remote deployment yet)
  - Compare runs in MLflow UI
- **Docker:**
  - Dockerfile for the training pipeline
  - docker-compose for MLflow + API together
  - Document the "it works on my machine" → Docker story in README
- **GitHub Actions CI:**
  - Lint (ruff)
  - Run tests
  - Train on push to main (with cached data)
- **Model artifacts:** save with metadata JSON (timestamp, config hash, metrics, git SHA)

**Month 2+ deepening:**
- Model registry (MLflow model registry or simple versioning scheme)
- Data drift detection (compare input distributions against training data)
- A/B testing scaffold
- Monitoring dashboard (prediction latency, error rates)

---

## 4. FastAPI Serving Layer

```python
# Minimal but professional
POST /predict          → single prediction (JSON in, JSON out)
POST /predict/batch    → batch predictions (CSV upload)
GET  /health           → healthcheck
GET  /model/info       → model metadata (version, metrics, training date)
```

**Request/response schema (Pydantic):**
```python
class PredictionRequest(BaseModel):
    Pclass: int
    Sex: str
    Age: float | None = None
    Fare: float
    SibSp: int = 0
    Parch: int = 0
    Embarked: str = "S"

class PredictionResponse(BaseModel):
    survived: bool
    probability: float
    model_version: str
```

---

## 5. Week-by-Week Roadmap

### Week 1: Foundation + Data Pipeline
- [ ] Project scaffolding (pyproject.toml, directory structure, git init)
- [ ] Config system (Pydantic models + YAML loading)
- [ ] Data loader with train/val/test splitting
- [ ] Exploratory notebook (EDA — this is the ONE notebook)
- [ ] First custom transformers (TitleExtractor, FamilySizeBuilder)
- [ ] Basic pytest setup with 3-5 unit tests
- **Milestone:** `ml-forge train --config configs/titanic.yaml` runs end-to-end with LogReg

### Week 2: Feature Engineering + Modeling Depth
- [ ] All custom transformers built and tested
- [ ] Feature selection comparison (mutual info, RFE, permutation importance)
- [ ] Full sklearn pipeline (ColumnTransformer + custom transformers + model)
- [ ] All base models trained and compared
- [ ] Optuna integration for top 2-3 models
- [ ] Stacking ensemble
- [ ] Evaluation suite (metrics, confusion matrix, learning curves)
- [ ] MLflow tracking for all experiments
- **Milestone:** Best model identified with rigorous comparison documented

### Week 3: Serving + Infrastructure
- [ ] FastAPI endpoints (predict, batch, health, model info)
- [ ] Dockerfile for training pipeline
- [ ] docker-compose (MLflow + FastAPI)
- [ ] Typer CLI polished (train, predict, evaluate, tune, serve)
- [ ] Structured logging throughout
- [ ] Integration test (full pipeline)
- [ ] Great Expectations basic suite
- **Milestone:** `docker-compose up` starts everything; API serves predictions

### Week 4: CI/CD + Polish
- [ ] GitHub Actions workflow (lint, test, train)
- [ ] README narrative (problem → approach → results → how to run)
- [ ] Model artifact versioning with metadata
- [ ] Code cleanup, docstrings, type hints throughout
- [ ] Final model performance summary with visualizations
- [ ] Record what you'd do in Month 2+ (shows growth mindset)
- **Milestone:** Someone can clone the repo, run `docker-compose up`, and get predictions

---

## 6. Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Language | Python 3.11+ | Industry standard for ML |
| ML | scikit-learn, XGBoost, LightGBM | Covers classical ML thoroughly |
| Tuning | Optuna | Modern, Bayesian, pruning support |
| Features | Custom sklearn transformers | Shows API design skill |
| Config | Pydantic + PyYAML | Validation + readability |
| CLI | Typer | Modern, type-hinted CLI framework |
| API | FastAPI | Async, auto-docs, Pydantic integration |
| Tracking | MLflow (local) | Industry standard experiment tracking |
| Testing | pytest + Great Expectations (light) | Sufficient for Month 1 |
| Logging | structlog | Structured, JSON-friendly |
| Container | Docker + docker-compose | Reproducibility |
| CI/CD | GitHub Actions | Already familiar |
| Linting | ruff | Fast, replaces flake8+isort+black |

---

## 7. Dataset-Agnostic Design

To swap in a new dataset (e.g., Heart Disease, Credit Default):

1. Create `configs/heart_disease.yaml` with column names, types, target
2. Write dataset-specific transformers in `features/` (if needed)
3. Run `ml-forge train --config configs/heart_disease.yaml`

The pipeline core (splitting, training, evaluation, tuning, serving) stays the same. This is the "kitchen vs. dish" principle — only the ingredients change.

---

## 8. README Narrative Structure

The README is the front door. Structure it as:

1. **What this is** (one paragraph — not "a Titanic classifier")
2. **Architecture diagram** (Mermaid or image)
3. **Quick start** (`docker-compose up` → API ready)
4. **Results** (best model, key metrics, what worked/didn't)
5. **Project structure** (annotated tree)
6. **How to add a new dataset** (shows extensibility)
7. **What I learned / What's next** (signals growth mindset)

---

## 9. Month 2-3 Extension Roadmap

- Deepen testing: Hypothesis property-based tests, >80% coverage, API contract tests
- MLOps maturity: model registry, drift detection, monitoring dashboard
- Second dataset: prove the "dataset-agnostic" claim
- Neural network baseline: simple MLP via PyTorch (bridge to deep learning)
- Paper-style writeup: formal analysis of results (PhD prep)