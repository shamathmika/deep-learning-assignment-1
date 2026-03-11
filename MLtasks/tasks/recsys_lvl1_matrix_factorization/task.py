"""
Neural Matrix Factorization (Neural MF)

Mathematical Formulation:
    Predicted rating:  r_hat(u, i) = μ + b_u + b_i + U[u] · V[i]
    where μ is a global mean, b_u / b_i are per-user / per-item biases,
    and U ∈ R^(n_users × k), V ∈ R^(n_items × k) are interaction embedding
    matrices — all learned end-to-end via backpropagation.

    MSE Loss:  L = (1/|R|) × Σ_{(u,i) ∈ R} (r_hat(u,i) - r(u,i))²

    Optimization: AdamW (torch.optim.AdamW) with decoupled weight decay.

This is the foundational deep-learning approach to collaborative filtering:
embeddings are a single trainable layer — the simplest possible neural network
for recommendation.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

def get_task_metadata():
    return {
        'task_id':     'recsys_lvl1_matrix_factorization',
        'series':      'Recommendation Systems',
        'level':       1,
        'algorithm':   'Neural Matrix Factorization',
        'description': (
            'User-item rating prediction using nn.Embedding matrices + bias terms. '
            'r_hat(u,i) = μ + b_u + b_i + U[u] · V[i]. '
            'MSE Loss: L = (1/|R|) × Σ (r_hat - r)². Trained with AdamW.'
        ),
    }


# ---------------------------------------------------------------------------
# 2. Seed
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# 3. Device
# ---------------------------------------------------------------------------

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# 4. Data
# ---------------------------------------------------------------------------

def make_dataloaders(cfg):
    """
    Generate a synthetic user-item rating dataset with latent-factor structure.
    200 users, 100 items, ~5% observed ratings in [1, 5], 80/20 train/val split.

    Returns (train_loader, val_loader) — standard PyTorch DataLoaders.
    """
    n_users   = cfg.get('n_users',    200)
    n_items   = cfg.get('n_items',    100)
    density   = cfg.get('density',   0.05)
    noise_std = cfg.get('noise_std',  0.4)
    seed      = cfg.get('seed',        42)
    batch     = cfg.get('batch_size',  64)
    k_true    = 5

    rng = np.random.RandomState(seed)

    # True latent factors → structured rating matrix
    U_true      = rng.randn(n_users, k_true) * 0.7
    V_true      = rng.randn(n_items, k_true) * 0.7
    true_scores = U_true @ V_true.T                          # (n_users, n_items)

    s_min, s_max = true_scores.min(), true_scores.max()
    true_ratings = (true_scores - s_min) / (s_max - s_min) * 4.0 + 1.0
    noisy = np.clip(true_ratings + rng.randn(n_users, n_items) * noise_std, 1.0, 5.0)

    # Sample observed (user, item) pairs
    n_obs     = int(n_users * n_items * density)
    all_pairs = [(u, i) for u in range(n_users) for i in range(n_items)]
    chosen    = rng.choice(len(all_pairs), size=n_obs, replace=False)
    obs       = [all_pairs[k] for k in chosen]

    users   = np.array([p[0] for p in obs])
    items   = np.array([p[1] for p in obs])
    ratings = np.array([noisy[u, i] for u, i in obs], dtype=np.float32)

    # 80/20 split
    perm    = rng.permutation(n_obs)
    n_train = int(n_obs * 0.8)
    tr, va  = perm[:n_train], perm[n_train:]

    def _loader(idx, shuffle):
        ds = TensorDataset(
            torch.tensor(users[idx],   dtype=torch.long),
            torch.tensor(items[idx],   dtype=torch.long),
            torch.tensor(ratings[idx], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    return _loader(tr, True), _loader(va, False)


# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------

class NeuralMFModel(nn.Module):
    """
    Neural Matrix Factorization with biases.

    r_hat(u, i) = global_mean + b_u + b_i + U[u] · V[i]

    U ∈ R^(n_users × embed_dim),  V ∈ R^(n_items × embed_dim)
    Bias terms absorb per-user/item offsets, keeping interaction embeddings small.
    All parameters trained end-to-end with AdamW.
    """

    def __init__(self, n_users, n_items, embed_dim=16):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, embed_dim)
        self.item_emb  = nn.Embedding(n_items, embed_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_mean = nn.Parameter(torch.tensor(3.0))
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        u   = self.user_emb(user_ids)
        v   = self.item_emb(item_ids)
        b_u = self.user_bias(user_ids).squeeze(-1)
        b_i = self.item_bias(item_ids).squeeze(-1)
        return self.global_mean + b_u + b_i + (u * v).sum(dim=-1)


def build_model(cfg):
    return NeuralMFModel(
        n_users   = cfg.get('n_users',   200),
        n_items   = cfg.get('n_items',   100),
        embed_dim = cfg.get('embed_dim',  16),
    )


# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------

def train(model, train_loader, cfg, device):
    """
    Standard epoch loop with AdamW + MSELoss.
    Returns list of per-epoch training losses.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.get('lr', 1e-3),
                                  weight_decay=cfg.get('weight_decay', 1e-2))
    criterion = nn.MSELoss()
    epochs    = cfg.get('epochs', 50)

    loss_history = []
    for epoch in range(epochs):
        total, n_bat = 0.0, 0
        for u_b, i_b, r_b in train_loader:
            u_b, i_b, r_b = u_b.to(device), i_b.to(device), r_b.to(device)
            optimizer.zero_grad()
            r_hat = model(u_b, i_b)
            loss  = criterion(r_hat, r_b)
            loss.backward()
            optimizer.step()
            total  += loss.item()
            n_bat  += 1
        avg = total / n_bat if n_bat > 0 else 0.0
        loss_history.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}]  MSE: {avg:.4f}")

    return loss_history


# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, loader, cfg, device):
    """
    Compute RMSE and MAE on a DataLoader.
    Returns dict: {'rmse': float, 'mae': float, 'mse': float}
    """
    model.to(device)
    model.eval()
    lo, hi = cfg.get('rating_min', 1.0), cfg.get('rating_max', 5.0)
    preds, targets = [], []
    with torch.no_grad():
        for u_b, i_b, r_b in loader:
            r_hat = model(u_b.to(device), i_b.to(device)).clamp(lo, hi)
            preds.append(r_hat.cpu())
            targets.append(r_b)
    preds   = torch.cat(preds)
    targets = torch.cat(targets)
    err     = preds - targets
    mse     = (err ** 2).mean().item()
    return {'rmse': round(mse ** 0.5, 4), 'mae': round(err.abs().mean().item(), 4),
            'mse': round(mse, 4)}


# ---------------------------------------------------------------------------
# 8. Predict
# ---------------------------------------------------------------------------

def predict(model, x, device):
    """
    Predict ratings for (user_ids, item_ids) pairs.

    Parameters
    ----------
    x : tuple (user_tensor, item_tensor)

    Returns
    -------
    1-D CPU tensor of predicted ratings clamped to [1, 5].
    """
    model.to(device)
    model.eval()
    user_ids, item_ids = x
    if not isinstance(user_ids, torch.Tensor):
        user_ids = torch.tensor(user_ids, dtype=torch.long)
        item_ids = torch.tensor(item_ids, dtype=torch.long)
    with torch.no_grad():
        out = model(user_ids.to(device), item_ids.to(device)).clamp(1.0, 5.0)
    return out.cpu()


# ---------------------------------------------------------------------------
# 9. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(outputs, cfg):
    """Save training loss curve."""
    out_dir = cfg.get('output_dir', 'outputs/recsys_lvl1_matrix_factorization')
    os.makedirs(out_dir, exist_ok=True)

    if outputs.get('loss_history'):
        plt.figure(figsize=(8, 4))
        plt.plot(outputs['loss_history'], color='steelblue')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Neural MF — Training Loss')
        plt.tight_layout()
        path = os.path.join(out_dir, 'recsys_lvl1_loss.png')
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Task : recsys_lvl1_matrix_factorization")
    print("Algo : Neural Matrix Factorization (nn.Embedding + Adam)")
    print("=" * 60)

    set_seed(42)
    device   = get_device()
    metadata = get_task_metadata()
    print(f"Device  : {device}")
    print(f"Task ID : {metadata['task_id']}")

    cfg = {
        'n_users':      200,
        'n_items':      100,
        'density':      0.05,
        'noise_std':    0.4,
        'embed_dim':     16,   # slightly above true rank (k_true=5); biases absorb offsets
        'lr':           2e-3,
        'weight_decay': 3e-2,  # AdamW decoupled L2 — moderate reg for sparse MF
        'epochs':       100,
        'batch_size':    64,
        'seed':          42,
        'output_dir':   'outputs/recsys_lvl1_matrix_factorization',
    }

    # 1 — Data
    print("\n[1] Creating dataset …")
    train_loader, val_loader = make_dataloaders(cfg)
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # 2 — Model
    print("\n[2] Building NeuralMFModel …")
    model   = build_model(cfg)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_param:,}")

    # 3 — Train
    print(f"\n[3] Training on {device} …")
    loss_history = train(model, train_loader, cfg, device)

    # 4 — Evaluate
    print("\n[4] Evaluating …")
    train_metrics = evaluate(model, train_loader, cfg, device)
    val_metrics   = evaluate(model, val_loader,   cfg, device)
    print(f"  Train  RMSE={train_metrics['rmse']:.4f}  MAE={train_metrics['mae']:.4f}")
    print(f"  Val    RMSE={val_metrics['rmse']:.4f}  MAE={val_metrics['mae']:.4f}")

    # 5 — Artifacts
    print("\n[5] Saving artifacts …")
    save_artifacts({'loss_history': loss_history}, cfg)

    # Sample predictions
    u_s = torch.tensor([0, 1, 2], dtype=torch.long)
    i_s = torch.tensor([0, 1, 2], dtype=torch.long)
    preds = predict(model, (u_s, i_s), device)
    print(f"\nSample predictions (u,i): {[round(v, 3) for v in preds.tolist()]}")

    # --- Assertions ---
    print("\n--- Assertions ---")
    ok = True

    if val_metrics['rmse'] < 1.0:
        print(f"PASS  val RMSE {val_metrics['rmse']:.4f} < 1.0")
    else:
        print(f"FAIL  val RMSE {val_metrics['rmse']:.4f} >= 1.0")
        ok = False

    if val_metrics['mae'] < 0.8:
        print(f"PASS  val MAE  {val_metrics['mae']:.4f} < 0.8")
    else:
        print(f"FAIL  val MAE  {val_metrics['mae']:.4f} >= 0.8")
        ok = False

    if ok:
        print("\nAll assertions passed.")
        sys.exit(0)
    else:
        print("\nAssertion failures detected.")
        sys.exit(1)
