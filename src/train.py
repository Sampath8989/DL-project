"""Probe training: single fit, temperature calibration, deep ensemble."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import build, count_params


def _standardize(train, *others):
    """Fit mean/std on the training split only, apply to all splits."""
    axes = tuple(range(train.ndim - 1))
    mu = train.mean(axis=axes, keepdims=True)
    sd = train.std(axis=axes, keepdims=True) + 1e-6
    out = [(train - mu) / sd] + [(o - mu) / sd for o in others]
    return out


def _loader(X, y, mask, bs, shuffle):
    ts = [torch.as_tensor(X, dtype=torch.float32),
          torch.as_tensor(y, dtype=torch.float32)]
    if mask is not None:
        ts.append(torch.as_tensor(mask, dtype=torch.float32))
    return DataLoader(TensorDataset(*ts), batch_size=bs, shuffle=shuffle)


def fit_one(Xtr, ytr, Xva, yva, cfg, seed=0, mask_tr=None, mask_va=None,
            device="cpu"):
    """Train one probe with early stopping on validation loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_model = Xtr.shape[-1]
    n_layers = Xtr.shape[1] if (cfg.arch == "layer_attn" and Xtr.ndim == 3) else None
    model = build(cfg.arch, d_model=d_model, n_layers=n_layers,
                  hidden=cfg.hidden, dropout=cfg.dropout).to(device)

    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    pw = torch.tensor([neg / max(pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    tr = _loader(Xtr, ytr, mask_tr, cfg.batch_size, True)
    va = _loader(Xva, yva, mask_va, cfg.batch_size, False)

    best, best_state, bad = float("inf"), None, 0
    for _ in range(cfg.epochs):
        model.train()
        for batch in tr:
            xb, yb = batch[0].to(device), batch[1].to(device)
            mb = batch[2].to(device) if len(batch) > 2 else None
            opt.zero_grad()
            loss = crit(model(xb, mb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in va:
                xb, yb = batch[0].to(device), batch[1].to(device)
                mb = batch[2].to(device) if len(batch) > 2 else None
                l = crit(model(xb, mb), yb)
                tot += float(l) * len(yb)
                n += len(yb)
        vl = tot / max(n, 1)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


def _logits(model, X, mask, device, bs=256):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
            mb = (torch.as_tensor(mask[i:i + bs], dtype=torch.float32, device=device)
                  if mask is not None else None)
            outs.append(model(xb, mb).cpu().numpy())
    return np.concatenate(outs)


def fit_temperature(logits, y, grid=None):
    """Single-parameter temperature scaling fitted on the validation split."""
    if grid is None:
        grid = np.linspace(0.25, 5.0, 96)
    y = np.asarray(y, dtype=float)
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        z = np.clip(logits / t, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def predict(model, X, temp=1.0, mask=None, device="cpu"):
    z = _logits(model, X, mask, device) / temp
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_ensemble(Xtr, ytr, Xva, yva, Xte, cfg, mask_tr=None, mask_va=None,
                 mask_te=None, device="cpu"):
    """Deep ensemble over cfg.seeds; returns mean probability and spread."""
    probs, members, temps = [], [], []
    for s in cfg.seeds:
        model, _ = fit_one(Xtr, ytr, Xva, yva, cfg, seed=s,
                           mask_tr=mask_tr, mask_va=mask_va, device=device)
        t = 1.0
        if cfg.calibrate:
            t = fit_temperature(_logits(model, Xva, mask_va, device), yva)
        probs.append(predict(model, Xte, t, mask_te, device))
        members.append(model)
        temps.append(t)
    P = np.stack(probs)
    return dict(prob=P.mean(0), spread=P.std(0), members=members, temps=temps,
                n_params=count_params(members[0]))
