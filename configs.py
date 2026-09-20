"""Configuration for the first-phase sectorized Tree-Ring experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class SectorConfig:
    # model / data
    model_id: str = "stabilityai/stable-diffusion-2-1-base"
    dataset: str = "Gustavosta/Stable-Diffusion-Prompts"
    image_length: int = 512
    num_inference_steps: int = 50
    test_num_inference_steps: int = 50
    guidance_scale: float = 7.5
    gen_seed: int = 0
    w_seed: int = 999999

    # sectorized carrier
    r1: int = 10
    r2: int = 20
    K: int = 16
    carrier: str = "walsh"          # const | rand | walsh | ortho
    channels: Tuple[int, ...] = (3,)  # latent channels carrying the message
    channel_mode: str = "diversity"   # diversity | capacity
    injection: str = "replace"        # replace | add
    n_pilot: int = 0
    n_guard: int = 0

    # energy budget
    energy_per_point: float = 1e4    # eta: mean per-point energy on annulus
    total_energy: Optional[float] = None  # explicit E_f, overrides eta

    # optional quality reference
    with_clip: bool = False
    reference_model: Optional[str] = None
    reference_model_pretrain: Optional[str] = None

    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    @property
    def K_ind(self) -> int:
        return self.K // 2

    @property
    def n_bits(self) -> int:
        """Payload bits per channel (excludes pilot/guard)."""
        return self.K_ind - self.n_pilot - self.n_guard

    @property
    def total_n_bits(self) -> int:
        if self.channel_mode == "capacity":
            return self.n_bits * len(self.channels)
        return self.n_bits


# experiment grids (doc sections 8.2, 8.3, 8.4, 8.5, 8.6, 8.7)
EXPERIMENTS = {
    "1": {
        "name": "feasibility",
        "configs": [
            dict(K=16, carrier="walsh"),
            dict(K=32, carrier="walsh"),
        ],
        "attacks": ["clean", "jpeg25", "noise0.1"],
    },
    "2": {
        "name": "capacity_inflection",
        "configs": [dict(K=k, carrier="walsh") for k in (16, 32, 48, 64)],
        "attacks": ["clean", "noise0.1"],
    },
    "3": {
        "name": "carriers",
        "configs": [
            dict(K=16, carrier=c)
            for c in ("const", "rand", "walsh", "ortho")
        ],
        "attacks": ["clean", "jpeg25", "noise0.1", "blur4", "bright1.5"],
    },
    "4": {
        "name": "energy",
        "configs": [
            dict(K=16, carrier="walsh", energy_per_point=e)
            for e in (1e2, 1e3, 1e4, 1e5)
        ],
        "attacks": ["clean", "jpeg25", "noise0.1"],
    },
    "5": {
        "name": "channels",
        "configs": [
            dict(K=16, carrier="walsh", channels=(3,), channel_mode="diversity"),
            dict(K=16, carrier="walsh", channels=(3, 2), channel_mode="diversity"),
            dict(K=16, carrier="walsh", channels=(3, 2, 0, 1), channel_mode="diversity"),
            dict(K=16, carrier="walsh", channels=(3, 2), channel_mode="capacity"),
            dict(K=16, carrier="walsh", channels=(3, 2, 0, 1), channel_mode="capacity"),
        ],
        "attacks": ["clean", "noise0.1"],
    },
    "7": {
        "name": "geometry_control",
        "configs": [dict(K=16, carrier="walsh")],
        "attacks": ["rot45", "rot75", "rot90", "crop0.75"],
    },
}
