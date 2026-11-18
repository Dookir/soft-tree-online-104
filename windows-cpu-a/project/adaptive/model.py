"""Three independent soft forests followed by a signed sum and L1 readout.

Latent channels are not supervised physical trajectories. No simulator assets
or pre-trained parameters are used by this module.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn


def helmert_basis() -> np.ndarray:
    basis = np.zeros((7, 6), dtype=np.float64)
    for j in range(1, 7):
        basis[:j, j - 1] = 1 / math.sqrt(j * (j + 1))
        basis[j, j - 1] = -j / math.sqrt(j * (j + 1))
    return basis


class DistrictSurrogate(nn.Module):
    """4 depth-3 trees per building, 32 channels; 3,692 parameters."""

    def __init__(self, scale: float, seed: int):
        super().__init__()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Target scale must be positive and finite")
        rng = np.random.default_rng(seed)
        self.gate_weight = nn.Parameter(torch.from_numpy(
            rng.normal(0, 1 / math.sqrt(6), (3, 4, 7, 6))))
        self.gate_bias = nn.Parameter(torch.from_numpy(rng.uniform(-.25, .25, (3, 4, 7))))
        self.leaves = nn.Parameter(torch.from_numpy(rng.normal(0, .7, (3, 4, 8, 32))))
        self.offset = nn.Parameter(torch.from_numpy(rng.normal(0, 1., 32)))
        self.register_buffer("basis", torch.from_numpy(helmert_basis()))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float64))
        nodes, rights = [], []
        for leaf in range(8):
            node, path, bits = 0, [], []
            for depth in range(3):
                bit = (leaf >> (2 - depth)) & 1
                path.append(node)
                bits.append(bit)
                node = 2 * node + 1 + bit
            nodes.append(path)
            rights.append(bits)
        self.register_buffer("path_nodes", torch.tensor(nodes, dtype=torch.long))
        self.register_buffer("path_right", torch.tensor(rights, dtype=torch.float64))

    def routing(self, coordinates: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid((torch.einsum("nbd,btrd->nbtr", coordinates,
                                          self.gate_weight) + self.gate_bias) / .2)
        probabilities = gates[..., self.path_nodes]
        return (probabilities * self.path_right +
                (1 - probabilities) * (1 - self.path_right)).prod(dim=-1)

    def forest(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nbtl,btlk->nbtk", self.routing(coordinates),
                            self.leaves).mean(dim=2)

    def branch_features(self, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.reshape(-1, 3, 7)
        # Inputs originate from a validated CSV or simplex projection. Dividing
        # by max first also handles finite but very large external weights.
        normalized = weights / weights.amax(dim=-1, keepdim=True)
        normalized = normalized / normalized.sum(dim=-1, keepdim=True)
        coordinates = normalized @ self.basis
        center = self.forest(coordinates.new_zeros((1, 3, 6)))
        return self.forest(coordinates) - center

    def normalized_prediction(self, weights: torch.Tensor) -> torch.Tensor:
        signed = self.branch_features(weights).sum(dim=1) + self.offset
        return signed.abs().mean(dim=-1)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        return self.scale * self.normalized_prediction(weights)
