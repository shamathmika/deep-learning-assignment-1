"""
Two-Tower Model with Retrieval Benchmark

Architecture:
    User tower : MLP(user_features ∈ R^8)  →  embedding ∈ R^64
    Item tower : MLP(item_features ∈ R^8)  →  embedding ∈ R^64
    Score      : dot product of L2-normalised embeddings

Loss (InfoNCE with in-batch negatives):
    For a batch of B (user_i, item_i+) pairs:
        logits[i, j] = score(user_i, item_j) / temperature
        L_i = -log( exp(logits[i,i] / τ) / Σ_j exp(logits[i,j] / τ) )
    L = CrossEntropyLoss(logits, target=diag)

Retrieval:
    Precompute all item embeddings at inference time.
    For each user query: scores = user_emb @ item_embs.T → top-K.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

def get_task_metadata():
    return {
        'task_id':     'recsys_lvl4_two_tower',
        'series':      'Recommendation Systems',
        'level':       4,
        'algorithm':   'Two-Tower Model with Retrieval Benchmark',
        'description': 'Separate user/item MLP towers with InfoNCE loss and in-batch '
                       'negatives. Precomputed item embeddings for fast retrieval.',
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
    1000 users (8 features) and 500 items (8 features).
    ~5% of user-item pairs are positive interactions.

    Returns (train_data, val_data) as dicts.
    """
    n_users  = cfg.get('n_users',  1000)
    n_items  = cfg.get('n_items',   500)
    feat_dim = cfg.get('feat_dim',    8)
    seed     = cfg.get('seed',       42)

    rng = np.random.RandomState(seed)

    user_feat = rng.randn(n_users, feat_dim).astype(np.float32)
    item_feat = rng.randn(n_items, feat_dim).astype(np.float32)

    scores    = user_feat @ item_feat.T
    threshold = np.percentile(scores, 95)           # top 5% → positive
    interaction_matrix = (scores >= threshold).astype(np.int32)
    print(f"  Interaction density: {interaction_matrix.mean():.3%}")

    train_pos, val_pos = {}, {}
    for u in range(n_users):
        pos_items = np.where(interaction_matrix[u] == 1)[0].tolist()
        if len(pos_items) < 2:
            continue
        rng.shuffle(pos_items)
        n_val        = max(1, int(len(pos_items) * 0.2))
        val_pos[u]   = set(pos_items[:n_val])
        train_pos[u] = set(pos_items[n_val:])

    user_feat_t = torch.tensor(user_feat)
    item_feat_t = torch.tensor(item_feat)

    train_data = {
        'user_features': user_feat_t, 'item_features': item_feat_t,
        'user_pos_items': train_pos, 'n_users': n_users, 'n_items': n_items,
    }
    val_data = {
        'user_features': user_feat_t, 'item_features': item_feat_t,
        'user_pos_items': val_pos, 'train_user_pos': train_pos,
        'n_users': n_users, 'n_items': n_items,
    }
    return train_data, val_data


# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------

def _mlp(in_dim, hidden_dims, out_dim):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TwoTowerModel(nn.Module):
    """
    Two separate MLP towers outputting L2-normalised 64-dim embeddings.

    user_tower(user_features) → u_emb ∈ R^64
    item_tower(item_features) → v_emb ∈ R^64
    score(u, i) = u_emb · v_emb
    """

    def __init__(self, feat_dim=8, hidden=(64, 64), embed_dim=64, temperature=0.07):
        super().__init__()
        self.user_tower  = _mlp(feat_dim, list(hidden), embed_dim)
        self.item_tower  = _mlp(feat_dim, list(hidden), embed_dim)
        self.temperature = temperature

    def encode_users(self, x):
        return F.normalize(self.user_tower(x), dim=-1)

    def encode_items(self, x):
        return F.normalize(self.item_tower(x), dim=-1)

    def forward(self, user_feat, item_feat):
        return self.encode_users(user_feat), self.encode_items(item_feat)


def build_model(cfg):
    return TwoTowerModel(
        feat_dim    = cfg.get('feat_dim',      8),
        hidden      = cfg.get('hidden',    (64, 64)),
        embed_dim   = cfg.get('embed_dim',    64),
        temperature = cfg.get('temperature', 0.07),
    )


# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------

def _build_train_loader(train_data, cfg):
    train_pos  = train_data['user_pos_items']
    user_feats = train_data['user_features']
    item_feats = train_data['item_features']

    us, ifs = [], []
    for u, pos_items in train_pos.items():
        for i in pos_items:
            us.append(u);  ifs.append(i)

    user_f = user_feats[torch.tensor(us, dtype=torch.long)]
    item_f = item_feats[torch.tensor(ifs, dtype=torch.long)]

    ds = TensorDataset(user_f, item_f)
    return DataLoader(ds, batch_size=cfg.get('batch_size', 256),
                      shuffle=True, drop_last=True)


def train(model, train_data, cfg, device):
    """
    InfoNCE training with in-batch negatives.

    Returns list of per-epoch losses.
    """
    model.to(device)
    loader    = _build_train_loader(train_data, cfg)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.get('lr', 1e-3),
                                 weight_decay=cfg.get('weight_decay', 1e-5))
    epochs       = cfg.get('epochs', 30)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total, n_bat = 0.0, 0
        for user_f, item_f in loader:
            user_f, item_f = user_f.to(device), item_f.to(device)
            u_emb, v_emb   = model(user_f, item_f)

            logits  = u_emb @ v_emb.T / model.temperature      # (B, B)
            targets = torch.arange(logits.size(0), device=device)
            loss    = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item();  n_bat += 1

        avg = total / max(n_bat, 1)
        loss_history.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}]  InfoNCE: {avg:.4f}")

    return loss_history


# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------

def _recall_ndcg(ranked, relevant, k=10):
    top_k  = ranked[:k]
    hits   = [1 if it in relevant else 0 for it in top_k]
    recall = sum(hits) / len(relevant) if relevant else 0.0
    dcg    = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
    idcg   = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return recall, (dcg / idcg if idcg > 0 else 0.0)


def evaluate(model, val_data, cfg, device):
    """
    Recall@10, NDCG@10, and retrieval latency (ms per query).

    Returns dict with 'recall_at_10', 'ndcg_at_10', 'latency_ms_per_query'.
    """
    model.to(device)
    model.eval()
    k = cfg.get('top_k', 10)

    # Precompute all item embeddings once
    with torch.no_grad():
        item_embs = model.encode_items(val_data['item_features'].to(device))  # (n_items, 64)

    user_feats = val_data['user_features'].to(device)
    val_pos    = val_data['user_pos_items']
    train_pos  = val_data.get('train_user_pos', {})

    recalls, ndcgs, latencies = [], [], []
    with torch.no_grad():
        for u, held_out in val_pos.items():
            if not held_out:
                continue
            t0     = time.perf_counter()
            u_emb  = model.encode_users(user_feats[u].unsqueeze(0))   # (1, 64)
            scores = (u_emb @ item_embs.T).squeeze(0).cpu().numpy()
            lat_ms = (time.perf_counter() - t0) * 1000

            for ti in train_pos.get(u, set()):
                scores[ti] = -1e9
            ranked = np.argsort(-scores).tolist()

            r, n = _recall_ndcg(ranked, held_out, k=k)
            recalls.append(r);  ndcgs.append(n);  latencies.append(lat_ms)

    return {
        'recall_at_10':         round(float(np.mean(recalls)),   4) if recalls   else 0.0,
        'ndcg_at_10':           round(float(np.mean(ndcgs)),     4) if ndcgs     else 0.0,
        'latency_ms_per_query': round(float(np.mean(latencies)), 3) if latencies else 0.0,
        'n_eval_users':         len(recalls),
    }


# ---------------------------------------------------------------------------
# 8. Predict
# ---------------------------------------------------------------------------

def predict(model, x, device):
    """
    Retrieve top-K items for user(s).

    Parameters
    ----------
    x : dict with keys 'user_features' and 'item_features'

    Returns
    -------
    Tensor (n_users, n_items) of dot-product scores.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        u_emb = model.encode_users(x['user_features'].to(device))
        v_emb = model.encode_items(x['item_features'].to(device))
        return (u_emb @ v_emb.T).cpu()


# ---------------------------------------------------------------------------
# 9. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(outputs, cfg):
    """Save InfoNCE loss curve and retrieval latency chart."""
    out_dir = cfg.get('output_dir', 'outputs/recsys_lvl4_two_tower')
    os.makedirs(out_dir, exist_ok=True)

    if outputs.get('loss_history'):
        plt.figure(figsize=(8, 4))
        plt.plot(outputs['loss_history'], color='steelblue')
        plt.xlabel('Epoch')
        plt.ylabel('InfoNCE Loss')
        plt.title('Two-Tower — Training Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'recsys_lvl4_loss.png'))
        plt.close()

    if outputs.get('latency_profile'):
        prof  = outputs['latency_profile']
        sizes = [p['n_items'] for p in prof]
        lats  = [p['ms'] for p in prof]
        plt.figure(figsize=(7, 4))
        plt.plot(sizes, lats, marker='o', color='steelblue')
        plt.xlabel('Number of Items')
        plt.ylabel('Latency (ms per query)')
        plt.title('Retrieval Latency vs Catalogue Size')
        plt.tight_layout()
        path = os.path.join(out_dir, 'recsys_lvl4_latency.png')
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Latency profile helper
# ---------------------------------------------------------------------------

def _latency_profile(model, val_data, device):
    """Measure retrieval latency at 100, 500, 1000 items."""
    model.to(device)
    model.eval()
    all_items  = val_data['item_features'].to(device)
    user_f     = val_data['user_features'][0].unsqueeze(0).to(device)
    profile    = []
    for n in [100, 500, min(1000, all_items.shape[0])]:
        item_sub = all_items[:n]
        times    = []
        for _ in range(25):
            t0 = time.perf_counter()
            with torch.no_grad():
                u = model.encode_users(user_f)
                v = model.encode_items(item_sub)
                _ = (u @ v.T)
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = float(np.mean(times[5:]))   # skip warm-up
        profile.append({'n_items': n, 'ms': round(avg_ms, 4)})
    return profile


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Task : recsys_lvl4_two_tower")
    print("Algo : Two-Tower Model (InfoNCE + in-batch negatives)")
    print("=" * 60)

    set_seed(42)
    device   = get_device()
    metadata = get_task_metadata()
    print(f"Device  : {device}")
    print(f"Task ID : {metadata['task_id']}")

    cfg = {
        'n_users':      1000,
        'n_items':       500,
        'feat_dim':        8,
        'hidden':      (64, 64),
        'embed_dim':      64,
        'temperature':  0.07,
        'lr':           1e-3,
        'weight_decay': 1e-5,
        'epochs':         30,
        'batch_size':    256,
        'top_k':          10,
        'seed':           42,
        'output_dir':    'outputs/recsys_lvl4_two_tower',
    }

    # 1 — Data
    print("\n[1] Creating dataset …")
    train_data, val_data = make_dataloaders(cfg)
    print(f"  Train users: {len(train_data['user_pos_items'])}")
    print(f"  Val   users: {len(val_data['user_pos_items'])}")

    # 2 — Build model
    print("\n[2] Building Two-Tower model …")
    model   = build_model(cfg)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_param:,}")

    # 3 — Train
    print(f"\n[3] Training on {device} …")
    loss_history = train(model, train_data, cfg, device)

    # 4 — Evaluate
    print("\n[4] Evaluating …")
    val_metrics = evaluate(model, val_data, cfg, device)
    print(f"  Recall@10       : {val_metrics['recall_at_10']:.4f}")
    print(f"  NDCG@10         : {val_metrics['ndcg_at_10']:.4f}")
    print(f"  Latency (ms/qry): {val_metrics['latency_ms_per_query']:.3f}")
    print(f"  Eval users      : {val_metrics['n_eval_users']}")

    # 5 — Latency profile
    print("\n[5] Retrieval latency at 100 / 500 / 1000 items …")
    lat_profile = _latency_profile(model, val_data, device)
    for p in lat_profile:
        print(f"  {p['n_items']:>5} items : {p['ms']:.4f} ms")

    # 6 — Artifacts
    print("\n[6] Saving artifacts …")
    save_artifacts({'loss_history': loss_history, 'latency_profile': lat_profile}, cfg)

    # Sample retrieval
    sample_input = {'user_features': val_data['user_features'][:3],
                    'item_features': val_data['item_features']}
    scores = predict(model, sample_input, device)
    top5   = scores[0].topk(5).indices.tolist()
    print(f"\nTop-5 items for user 0: {top5}")

    # --- Assertions ---
    print("\n--- Assertions ---")
    ok = True

    if val_metrics['recall_at_10'] > 0.05:
        print(f"PASS  Recall@10 {val_metrics['recall_at_10']:.4f} > 0.05")
    else:
        print(f"FAIL  Recall@10 {val_metrics['recall_at_10']:.4f} <= 0.05")
        ok = False

    max_lat = max(p['ms'] for p in lat_profile)
    if max_lat < 100.0:
        print(f"PASS  Max latency {max_lat:.3f}ms < 100ms")
    else:
        print(f"FAIL  Max latency {max_lat:.3f}ms >= 100ms")
        ok = False

    if ok:
        print("\nAll assertions passed.")
        sys.exit(0)
    else:
        print("\nAssertion failures detected.")
        sys.exit(1)
