"""Configuration objects for data, extraction and probe training.

Every field here is serialised into the result json, so a run can be
reconstructed from its output alone.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Which evaluation set to build, and how to sample it."""
    source: str = "popqa"          # popqa | triviaqa | contamination
    n: Optional[int] = 800
    seed: int = 0
    stratify: bool = True          # even coverage across popularity bands


@dataclass
class ExtractConfig:
    """Controls the frozen-LM forward pass and what gets cached to disk."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "float16"
    device: str = "cuda"
    max_new_tokens: int = 32
    max_items: Optional[int] = None
    batch_size: int = 1
    keep_token_trajectory: bool = True
    max_traj_tokens: int = 16
    layer_stride: int = 1          # raise to 2 or 4 to shrink the traj cache
    out_dir: str = "out/acts"


@dataclass
class ProbeConfig:
    """Controls probe architecture, optimisation and the ensemble."""
    arch: str = "layer_attn"       # linear | mlp | layer_attn | traj
    hidden: int = 128
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 60
    patience: int = 10
    batch_size: int = 64
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    standardize: bool = True
    calibrate: bool = True
