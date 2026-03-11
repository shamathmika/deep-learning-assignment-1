# CoderGym
Curriculum-style ML tasks with specs, tests, and artifacts for training and evaluating code LLMs.

---

## CMPE 258 — Homework 1: Deep Learning Training and Evaluation

Four new deep learning tasks were added under the **Recommendation Systems** series, following the `pytorch_task_v1` protocol. All tasks are implemented in PyTorch, use `nn.Module` subclasses, and are self-verifiable via `sys.exit(exit_code)` (exit 0 on pass, exit 1 on fail).

Each task file contains the 9 required protocol functions (`get_task_metadata`, `set_seed`, `get_device`, `make_dataloaders`, `build_model`, `train`, `evaluate`, `predict`, `save_artifacts`) plus a `__main__` block that trains, evaluates, and asserts quality thresholds.

### How to Run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib

# Run all 4 tasks
python MLtasks/tasks/recsys_lvl1_matrix_factorization/task.py
python MLtasks/tasks/recsys_lvl2_ncf/task.py
python MLtasks/tasks/recsys_lvl3_bpr/task.py
python MLtasks/tasks/recsys_lvl4_two_tower/task.py
```

Each script prints training progress, evaluation metrics, and assertion results. Output plots are saved to `outputs/`.

---

### Task 1 — Neural Matrix Factorization

**File:** `MLtasks/tasks/recsys_lvl1_matrix_factorization/task.py`

**Model:** `NeuralMFModel(nn.Module)` — `nn.Embedding` layers for user/item interactions plus per-user and per-item bias terms.

**Formula:** `r_hat(u, i) = μ + b_u + b_i + U[u] · V[i]`

**Loss:** MSE · **Optimizer:** AdamW with decoupled weight decay · **Data:** 200 users, 100 items, ~5% density, synthetic latent factor structure

**Results:**
| Metric | Value | Threshold |
|--------|-------|-----------|
| Val RMSE | 0.6311 | < 1.0 |
| Val MAE | 0.4870 | < 0.8 |

**Artifact:** `outputs/recsys_lvl1_matrix_factorization/recsys_lvl1_loss.png`

---

### Task 2 — Neural Collaborative Filtering (NCF)

**File:** `MLtasks/tasks/recsys_lvl2_ncf/task.py`

**Model:** `NCFModel(nn.Module)` — User and item embeddings concatenated and passed through a 3-layer MLP (128 → 64 → 32 → 1) with ReLU and Dropout. Compared against a `DotProductBaseline` to demonstrate the MLP's ability to capture nonlinear interaction patterns.

**Loss:** MSE · **Optimizer:** Adam · **Data:** 500 users, 200 items, ~5% density, with user/item biases in the data generation

**Results:**
| Metric | Value | Threshold |
|--------|-------|-----------|
| NCF Val RMSE | 0.4766 | < 1.0 |
| Baseline Val RMSE | 0.5537 | NCF must beat it |

**Artifact:** `outputs/recsys_lvl2_ncf/recsys_lvl2_loss.png`

---

### Task 3 — Bayesian Personalized Ranking (BPR)

**File:** `MLtasks/tasks/recsys_lvl3_bpr/task.py`

**Model:** `BPRModel(nn.Module)` — `nn.Embedding` layers with a 2-layer MLP scorer (128 → 64 → 1) applied on concatenated user-item embeddings. Trained with pairwise BPR loss using triplet sampling (user, positive item, negative item).

**Loss:** `−mean(log σ(score_pos − score_neg))` · **Optimizer:** Adam · **Data:** 500 users, 300 items, ~3% implicit feedback

**Results:**
| Metric | Value | Threshold |
|--------|-------|-----------|
| Recall@10 | 0.1364 | > 0.05 |
| NDCG@10 | 0.0684 | > 0.03 |

**Artifact:** `outputs/recsys_lvl3_bpr/recsys_lvl3_metrics.png`

---

### Task 4 — Two-Tower Model with InfoNCE

**File:** `MLtasks/tasks/recsys_lvl4_two_tower/task.py`

**Model:** `TwoTowerModel(nn.Module)` — Two separate MLP towers (8 → 64 → 64 → 64) encode user and item feature vectors into a shared L2-normalized embedding space. Item embeddings are precomputed at inference time for fast approximate nearest neighbor retrieval.

**Loss:** InfoNCE with in-batch negatives (`CrossEntropyLoss(sim_matrix / τ, diag_targets)`) · **Optimizer:** Adam · **Data:** 1000 users, 500 items, 8 features each, ~5% implicit feedback

**Results:**
| Metric | Value | Threshold |
|--------|-------|-----------|
| Recall@10 | 0.7509 | > 0.05 |
| NDCG@10 | 0.6954 | — |
| Max retrieval latency | 0.163 ms | < 100 ms |

**Artifact:** `outputs/recsys_lvl4_two_tower/recsys_lvl4_latency.png`

---

### Output Artifacts

All tasks produce training/evaluation plots saved under `outputs/`:

```
outputs/
├── recsys_lvl1_matrix_factorization/
│   └── recsys_lvl1_loss.png
├── recsys_lvl2_ncf/
│   └── recsys_lvl2_loss.png
├── recsys_lvl3_bpr/
│   └── recsys_lvl3_metrics.png
└── recsys_lvl4_two_tower/
    ├── recsys_lvl4_latency.png
    └── recsys_lvl4_loss.png
```
