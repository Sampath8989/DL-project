"""Four probe architectures of increasing capacity.

All probes consume frozen hidden states from a language model and emit a
single logit: p(the generated answer is hallucinated).

Input ranks
    LinearProbe, MLPProbe   x : [B, d]          one chosen layer
    LayerAttentionProbe     x : [B, L, d]       every layer
    TrajectoryProbe         x : [B, T, d]       one layer, T generated tokens
"""
import torch
import warnings
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
import torch.nn as nn


class LinearProbe(nn.Module):
    """Capacity floor. Tests whether truth is linearly readable."""

    def __init__(self, d_model, **_):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        return self.fc(x).squeeze(-1)


class MLPProbe(nn.Module):
    """Non-linear readout from a single layer."""

    def __init__(self, d_model, hidden=128, dropout=0.2, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, mask=None):
        return self.net(x).squeeze(-1)


class LayerAttentionProbe(nn.Module):
    """Learns which depths carry the signal instead of fixing one layer.

    A single learned query attends over the layer axis; the resulting
    attention weights are retained (``last_attn``) and reported as a
    depth profile, which is an interpretable output in its own right.
    """

    def __init__(self, d_model, n_layers, hidden=128, dropout=0.2, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden))
        self.layer_emb = nn.Parameter(torch.zeros(n_layers, hidden))
        nn.init.normal_(self.layer_emb, std=0.02)
        self.query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.scale = hidden ** -0.5
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.last_attn = None

    def forward(self, x, mask=None):
        h = self.proj(x) + self.layer_emb          # [B, L, H]
        scores = (h @ self.query) * self.scale     # [B, L]
        attn = torch.softmax(scores, dim=-1)
        self.last_attn = attn.detach()
        pooled = torch.einsum("bl,blh->bh", attn, h)
        return self.head(self.drop(pooled)).squeeze(-1)


class TrajectoryProbe(nn.Module):
    """Reads the whole generated-token trajectory with a small Transformer.

    Tests whether hallucination is visible in how the state *evolves*
    during generation rather than in any single pooled vector.
    """

    def __init__(self, d_model, hidden=128, dropout=0.2, depth=2, nhead=4, **_):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden))
        self.cls = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        B = x.shape[0]
        h = self.inp(x)                                       # [B, T, H]
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)  # [B, 1+T, H]
        if mask is not None:
            pad = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device),
                             mask.bool()], dim=1)
            h = self.enc(h, src_key_padding_mask=~pad)
        else:
            h = self.enc(h)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


REGISTRY = {
    "linear": LinearProbe,
    "mlp": MLPProbe,
    "layer_attn": LayerAttentionProbe,
    "traj": TrajectoryProbe,
}


def build(arch, **kwargs):
    if arch not in REGISTRY:
        raise KeyError(f"unknown probe arch {arch!r}; have {sorted(REGISTRY)}")
    return REGISTRY[arch](**kwargs)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
