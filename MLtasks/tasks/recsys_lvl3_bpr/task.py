"""
Bayesian Personalized Ranking (BPR) with Implicit Feedback

Setting:
    Implicit feedback — binary matrix (1 = interacted, 0 = not observed).
    500 users, 300 items, ~3% positive interactions.

Scoring (neural):
    score(u, i) = MLP( concat(U[u], V[i]) )
    where U, V are nn.Embedding layers and MLP is a 2-layer network.

BPR Loss (pairwise, per triplet (u, i+, i-)):
    L_BPR = -mean( log σ(score(u, i+) - score(u, i-)) )
    where i+ is a positive item and i- is a randomly sampled negative item.

Evaluation (ranking):
    Recall@10 = |Relevant ∩ Top-10| / |Relevant|
    NDCG@10   = DCG@10 / IDCG@10
                DCG@10  = Σ_{k=1}^{10} rel_k / log2(k+1)
                IDCG@10 = Σ_{k=1}^{|Rel|≤10} 1 / log2(k+1)
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


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

def get_task_metadata():
    return {
        'task_id':     'recsys_lvl3_bpr',
        'series':      'Recommendation Systems',
        'level':       3,
        'algorithm':   'Bayesian Personalized Ranking (BPR) with Implicit Feedback',
        'description': 'Triplet-based BPR loss on implicit feedback. '
                       '2-layer MLP scorer on concatenated embeddings. '
                       'Evaluate Recall@10 and NDCG@10.',
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
    Synthetic implicit feedback with latent structure.
    500 users, 300 items, ~3% positive interactions.

    Per-user: 80% positives → train, 20% (≥1) → validation.
    Users with < 2 positives are excluded from evaluation.

    Returns (train_data, val_data) as dicts.
    """
    n_users = cfg.get('n_users', 500)
    n_items = cfg.get('n_items', 300)
    seed    = cfg.get('seed',     42)
    k_true  = 8

    rng = np.random.RandomState(seed)

    U_true = rng.randn(n_users, k_true) * 0.6
    V_true = rng.randn(n_items, k_true) * 0.6
    scores = U_true @ V_true.T
    threshold = np.percentile(scores, 97)           # top 3% → positive
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

    train_data = {'user_pos_items': train_pos, 'n_users': n_users, 'n_items': n_items}
    val_data   = {'user_pos_items': val_pos, 'train_user_pos': train_pos,
                  'n_users': n_users, 'n_items': n_items}
    return train_data, val_data


# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------

class BPRModel(nn.Module):
    """
    Neural BPR scorer.

    score(u, i) = MLP( concat(U[u], V[i]) )

    Architecture:
        nn.Embedding  →  concat  →  Linear(128) → ReLU → Linear(64) → ReLU → Linear(1)
    """

    def __init__(self, n_users, n_items, embed_dim=64, mlp_hidden=128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.scorer   = nn.Sequential(
            nn.Linear(2 * embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def _score_pairs(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        return self.scorer(torch.cat([u, v], dim=-1)).squeeze(-1)

    def forward(self, user_ids, pos_ids, neg_ids):
        """Return (r_pos, r_neg) scores for BPR loss computation."""
        return self._score_pairs(user_ids, pos_ids), self._score_pairs(user_ids, neg_ids)

    def score_all_items(self, user_id, n_items, device):
        """Score a single user against all items (batched through MLP)."""
        u_emb  = self.user_emb(torch.tensor([user_id], device=device))  # (1, k)
        u_rep  = u_emb.expand(n_items, -1)                               # (n_items, k)
        i_ids  = torch.arange(n_items, device=device)
        v_emb  = self.item_emb(i_ids)                                    # (n_items, k)
        pairs  = torch.cat([u_rep, v_emb], dim=-1)                       # (n_items, 2k)
        return self.scorer(pairs).squeeze(-1)                             # (n_items,)


def build_model(cfg):
    return BPRModel(
        n_users    = cfg.get('n_users',     500),
        n_items    = cfg.get('n_items',     300),
        embed_dim  = cfg.get('embed_dim',    64),
        mlp_hidden = cfg.get('mlp_hidden',  128),
    )


# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------

def _sample_triplets(train_pos, n_users, n_items, n_triplets, rng):
    eligible   = [u for u, items in train_pos.items() if len(items) > 0]
    all_items  = set(range(n_items))
    users, pos_items, neg_items = [], [], []
    for _ in range(n_triplets):
        u        = rng.choice(eligible)
        pos      = rng.choice(list(train_pos[u]))
        neg      = rng.choice(list(all_items - train_pos[u]))
        users.append(u);  pos_items.append(pos);  neg_items.append(neg)
    return (torch.tensor(users,     dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long))


def train(model, train_data, cfg, device):
    """
    BPR training with triplet sampling.

    L_BPR = -mean( log σ(score(u,i+) - score(u,i-)) )

    Returns list of per-epoch BPR losses.
    """
    model.to(device)
    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=cfg.get('lr', 1e-3),
                                  weight_decay=cfg.get('weight_decay', 1e-5))
    epochs     = cfg.get('epochs',      50)
    n_triplets = cfg.get('n_triplets', 2048)
    train_pos  = train_data['user_pos_items']
    n_users    = train_data['n_users']
    n_items    = train_data['n_items']
    rng        = np.random.RandomState(cfg.get('seed', 42))

    loss_history = []
    for epoch in range(epochs):
        model.train()
        users, pos_items, neg_items = _sample_triplets(
            train_pos, n_users, n_items, n_triplets, rng)
        users, pos_items, neg_items = (users.to(device),
                                       pos_items.to(device),
                                       neg_items.to(device))
        optimizer.zero_grad()
        r_pos, r_neg = model(users, pos_items, neg_items)
        loss = -torch.log(torch.sigmoid(r_pos - r_neg) + 1e-8).mean()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}]  BPR loss: {loss.item():.4f}")

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
    Compute Recall@10 and NDCG@10 on held-out interactions.

    Returns dict: {'recall_at_10': float, 'ndcg_at_10': float}
    """
    model.to(device)
    model.eval()
    k         = cfg.get('top_k', 10)
    val_pos   = val_data['user_pos_items']
    train_pos = val_data.get('train_user_pos', {})
    n_items   = val_data['n_items']

    recalls, ndcgs = [], []
    with torch.no_grad():
        for u, held_out in val_pos.items():
            if not held_out:
                continue
            scores = model.score_all_items(u, n_items, device).cpu().numpy()
            for ti in train_pos.get(u, set()):
                scores[ti] = -1e9
            ranked = np.argsort(-scores).tolist()
            r, n   = _recall_ndcg(ranked, held_out, k=k)
            recalls.append(r);  ndcgs.append(n)

    return {
        'recall_at_10': round(float(np.mean(recalls)), 4) if recalls else 0.0,
        'ndcg_at_10':   round(float(np.mean(ndcgs)),   4) if ndcgs   else 0.0,
        'n_eval_users': len(recalls),
    }


# ---------------------------------------------------------------------------
# 8. Predict
# ---------------------------------------------------------------------------

def predict(model, x, device):
    """
    Score all items for user(s).

    Parameters
    ----------
    x : int or 1-D LongTensor of user ids

    Returns
    -------
    Tensor (n_users, n_items) of raw scores.
    """
    model.to(device)
    model.eval()
    if isinstance(x, int):
        x = torch.tensor([x], dtype=torch.long)
    n_items = model.item_emb.num_embeddings
    scores  = []
    with torch.no_grad():
        for uid in x:
            scores.append(model.score_all_items(uid.item(), n_items, device).cpu())
    return torch.stack(scores)


# ---------------------------------------------------------------------------
# 9. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(outputs, cfg):
    """Save BPR loss curve."""
    out_dir = cfg.get('output_dir', 'outputs/recsys_lvl3_bpr')
    os.makedirs(out_dir, exist_ok=True)

    if outputs.get('loss_history'):
        plt.figure(figsize=(8, 4))
        plt.plot(outputs['loss_history'], color='steelblue')
        plt.xlabel('Epoch')
        plt.ylabel('BPR Loss')
        plt.title('BPR — Training Loss')
        plt.tight_layout()
        path = os.path.join(out_dir, 'recsys_lvl3_metrics.png')
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Task : recsys_lvl3_bpr")
    print("Algo : BPR — Neural Embedding + 2-layer MLP Scorer")
    print("=" * 60)

    set_seed(42)
    device   = get_device()
    metadata = get_task_metadata()
    print(f"Device  : {device}")
    print(f"Task ID : {metadata['task_id']}")

    cfg = {
        'n_users':      500,
        'n_items':      300,
        'embed_dim':     64,
        'mlp_hidden':   128,
        'lr':           1e-3,
        'weight_decay': 1e-5,
        'epochs':        50,
        'n_triplets':  2048,
        'top_k':         10,
        'seed':          42,
        'output_dir':   'outputs/recsys_lvl3_bpr',
    }

    # 1 — Data
    print("\n[1] Creating implicit feedback dataset …")
    train_data, val_data = make_dataloaders(cfg)
    print(f"  Train users: {len(train_data['user_pos_items'])}")
    print(f"  Val   users: {len(val_data['user_pos_items'])}")

    # 2 — Build model
    print("\n[2] Building BPR model …")
    model   = build_model(cfg)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_param:,}")

    # 3 — Train
    print(f"\n[3] Training on {device} …")
    loss_history = train(model, train_data, cfg, device)

    # 4 — Evaluate
    print("\n[4] Evaluating …")
    val_metrics = evaluate(model, val_data, cfg, device)
    print(f"  Recall@10 : {val_metrics['recall_at_10']:.4f}")
    print(f"  NDCG@10   : {val_metrics['ndcg_at_10']:.4f}")
    print(f"  Eval users: {val_metrics['n_eval_users']}")

    # 5 — Artifacts
    print("\n[5] Saving artifacts …")
    save_artifacts({'loss_history': loss_history}, cfg)

    # Sample predictions
    scores = predict(model, 0, device)
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

    if val_metrics['ndcg_at_10'] > 0.03:
        print(f"PASS  NDCG@10   {val_metrics['ndcg_at_10']:.4f} > 0.03")
    else:
        print(f"FAIL  NDCG@10   {val_metrics['ndcg_at_10']:.4f} <= 0.03")
        ok = False

    if ok:
        print("\nAll assertions passed.")
        sys.exit(0)
    else:
        print("\nAssertion failures detected.")
        sys.exit(1)
