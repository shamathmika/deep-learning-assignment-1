"""
Neural Collaborative Filtering (NCF)

Architecture:
    User embedding  : nn.Embedding(n_users, embed_dim)
    Item embedding  : nn.Embedding(n_items, embed_dim)
    Concatenation   : [user_emb | item_emb]  →  (2 × embed_dim,)
    MLP (3 layers)  : Linear → ReLU → Linear → ReLU → Linear → ReLU → Linear(1)
    Output          : scalar predicted rating (clamped to [1, 5])

Loss      : MSELoss
Optimizer : Adam

Dot-product baseline: r_hat(u,i) = U[u] · V[i]  (no MLP, same optimizer)
NCF must outperform the baseline on validation RMSE.
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
        'task_id':     'recsys_lvl2_ncf',
        'series':      'Recommendation Systems',
        'level':       2,
        'algorithm':   'Neural Collaborative Filtering (NCF)',
        'description': 'MLP over concatenated user/item embeddings. Adam optimizer. '
                       'NCF is compared against a dot-product baseline.',
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
    Synthetic ratings for 500 users × 200 items, ~5% observed.
    Ratings generated with latent factor structure + per-user/item biases
    so the MLP has a natural advantage over a plain dot-product.

    Returns (train_loader, val_loader) — standard PyTorch DataLoaders.
    """
    n_users   = cfg.get('n_users',    500)
    n_items   = cfg.get('n_items',    200)
    density   = cfg.get('density',   0.05)
    noise_std = cfg.get('noise_std', 0.35)
    seed      = cfg.get('seed',        42)
    k_true    = 5
    batch     = cfg.get('batch_size', 256)

    rng = np.random.RandomState(seed)

    U_true      = rng.randn(n_users, k_true) * 0.6
    V_true      = rng.randn(n_items, k_true) * 0.6
    user_bias   = rng.randn(n_users) * 0.5
    item_bias   = rng.randn(n_items) * 0.5
    true_scores = U_true @ V_true.T + user_bias[:, None] + item_bias[None, :]

    s_min, s_max = true_scores.min(), true_scores.max()
    true_ratings = (true_scores - s_min) / (s_max - s_min) * 4.0 + 1.0
    noisy = np.clip(true_ratings + rng.randn(n_users, n_items) * noise_std, 1.0, 5.0)

    n_obs     = int(n_users * n_items * density)
    all_pairs = [(u, i) for u in range(n_users) for i in range(n_items)]
    chosen    = rng.choice(len(all_pairs), size=n_obs, replace=False)
    obs       = [all_pairs[k] for k in chosen]

    users   = np.array([p[0] for p in obs])
    items   = np.array([p[1] for p in obs])
    ratings = np.array([noisy[u, i] for u, i in obs], dtype=np.float32)

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

class NCFModel(nn.Module):
    """Neural Collaborative Filtering: concat embeddings → 3-layer MLP."""

    def __init__(self, n_users, n_items, embed_dim=32, mlp_dims=(128, 64, 32)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        layers = []
        in_dim = 2 * embed_dim
        for h in mlp_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        return self.mlp(torch.cat([u, v], dim=-1)).squeeze(-1)


class DotProductBaseline(nn.Module):
    """Baseline: r_hat(u,i) = U[u] · V[i]  (no MLP)."""

    def __init__(self, n_users, n_items, embed_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        return (self.user_emb(user_ids) * self.item_emb(item_ids)).sum(dim=-1)


def build_model(cfg):
    return NCFModel(
        n_users   = cfg.get('n_users',       500),
        n_items   = cfg.get('n_items',       200),
        embed_dim = cfg.get('embed_dim',      32),
        mlp_dims  = cfg.get('mlp_dims', (128, 64, 32)),
    )


# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------

def train(model, train_loader, cfg, device):
    """Train with Adam + MSELoss. Returns list of per-epoch losses."""
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.get('lr', 1e-3),
                                 weight_decay=cfg.get('weight_decay', 1e-5))
    criterion = nn.MSELoss()
    epochs    = cfg.get('epochs', 50)

    loss_history = []
    for epoch in range(epochs):
        total, n_bat = 0.0, 0
        for u_b, i_b, r_b in train_loader:
            u_b, i_b, r_b = u_b.to(device), i_b.to(device), r_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(u_b, i_b), r_b)
            loss.backward()
            optimizer.step()
            total  += loss.item()
            n_bat  += 1
        avg = total / n_bat if n_bat > 0 else 0.0
        loss_history.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}]  MSE: {avg:.4f}")

    return loss_history


def _train_baseline(model, train_loader, cfg, device):
    """Train the dot-product baseline with the same settings."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.get('lr', 1e-3),
                                 weight_decay=cfg.get('weight_decay', 1e-5))
    criterion = nn.MSELoss()
    for _ in range(cfg.get('epochs', 50)):
        for u_b, i_b, r_b in train_loader:
            u_b, i_b, r_b = u_b.to(device), i_b.to(device), r_b.to(device)
            optimizer.zero_grad()
            criterion(model(u_b, i_b), r_b).backward()
            optimizer.step()


# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, loader, cfg, device):
    """Compute RMSE and MAE. Returns dict."""
    model.to(device)
    model.eval()
    lo, hi  = cfg.get('rating_min', 1.0), cfg.get('rating_max', 5.0)
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
    Predict ratings.

    Parameters
    ----------
    x : tuple (user_tensor, item_tensor)
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
    out_dir = cfg.get('output_dir', 'outputs/recsys_lvl2_ncf')
    os.makedirs(out_dir, exist_ok=True)

    if outputs.get('loss_history'):
        plt.figure(figsize=(8, 4))
        plt.plot(outputs['loss_history'], color='steelblue', label='NCF Train MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('NCF — Training Loss')
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, 'recsys_lvl2_loss.png')
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Task : recsys_lvl2_ncf")
    print("Algo : Neural Collaborative Filtering (NCF)")
    print("=" * 60)

    set_seed(42)
    device   = get_device()
    metadata = get_task_metadata()
    print(f"Device  : {device}")
    print(f"Task ID : {metadata['task_id']}")

    cfg = {
        'n_users':       500,
        'n_items':       200,
        'density':       0.05,
        'noise_std':     0.35,
        'embed_dim':      32,
        'mlp_dims':      (128, 64, 32),
        'lr':            1e-3,
        'weight_decay':  1e-5,
        'epochs':         50,
        'batch_size':    256,
        'rating_min':    1.0,
        'rating_max':    5.0,
        'seed':           42,
        'output_dir':    'outputs/recsys_lvl2_ncf',
    }

    # 1 — Data
    print("\n[1] Creating dataset …")
    train_loader, val_loader = make_dataloaders(cfg)
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # 2 — Build NCF
    print("\n[2] Building NCF model …")
    model   = build_model(cfg)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_param:,}")

    # 3 — Train NCF
    print(f"\n[3] Training NCF on {device} …")
    loss_history = train(model, train_loader, cfg, device)

    # 4 — Evaluate NCF
    print("\n[4] Evaluating NCF …")
    val_metrics = evaluate(model, val_loader, cfg, device)
    print(f"  NCF  Val RMSE: {val_metrics['rmse']:.4f}  MAE: {val_metrics['mae']:.4f}")

    # 5 — Train & evaluate dot-product baseline
    print("\n[5] Training dot-product baseline …")
    baseline         = DotProductBaseline(cfg['n_users'], cfg['n_items'], cfg['embed_dim'])
    _train_baseline(baseline, train_loader, cfg, device)
    baseline_metrics = evaluate(baseline, val_loader, cfg, device)
    print(f"  Base Val RMSE: {baseline_metrics['rmse']:.4f}  MAE: {baseline_metrics['mae']:.4f}")

    # 6 — Artifacts
    print("\n[6] Saving artifacts …")
    save_artifacts({'loss_history': loss_history}, cfg)

    # Sample predictions
    u_s   = torch.tensor([0, 1, 2], dtype=torch.long)
    i_s   = torch.tensor([0, 1, 2], dtype=torch.long)
    preds = predict(model, (u_s, i_s), device)
    print(f"\nSample predictions: {[round(v, 3) for v in preds.tolist()]}")

    # --- Assertions ---
    print("\n--- Assertions ---")
    ok = True

    if val_metrics['rmse'] < 1.0:
        print(f"PASS  NCF val RMSE {val_metrics['rmse']:.4f} < 1.0")
    else:
        print(f"FAIL  NCF val RMSE {val_metrics['rmse']:.4f} >= 1.0")
        ok = False

    if val_metrics['rmse'] < baseline_metrics['rmse']:
        print(f"PASS  NCF ({val_metrics['rmse']:.4f}) beats baseline ({baseline_metrics['rmse']:.4f})")
    else:
        print(f"FAIL  NCF ({val_metrics['rmse']:.4f}) does not beat baseline ({baseline_metrics['rmse']:.4f})")
        ok = False

    if ok:
        print("\nAll assertions passed.")
        sys.exit(0)
    else:
        print("\nAssertion failures detected.")
        sys.exit(1)
