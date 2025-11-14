"""
Simplified Vector Quantization (SimVQ)

SimVQ uses an implicit neural codebook derived from a frozen random codebook
through a learnable transformation. This approach simplifies training while
maintaining quantization quality.

Key idea:
- Start with frozen random codebook
- Learn a transformation (typically a neural network) that maps frozen codes to actual codes
- Avoids complex codebook update mechanisms (EMA, k-means, etc.)
- Works well with rotation trick for gradient flow
"""

from __future__ import annotations
from typing import Callable

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einx import get_at
from einops import rearrange, pack, unpack

from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to

# ===================================
# Helper Functions
# ===================================

def exists(v):
    """Check if a value is not None"""
    return v is not None

def identity(t):
    """Identity function - returns input unchanged"""
    return t

def default(v, d):
    """Return v if it exists, otherwise return default value d"""
    return v if exists(v) else d

def pack_one(t, pattern):
    """
    Pack a single tensor and return an inverse function to unpack.
    This is useful for temporarily flattening tensors for processing.
    """
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out, = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse

# ===================================
# Simplified Vector Quantization
# ===================================

class SimVQ(Module):
    """
    Simplified Vector Quantization with implicit neural codebook.

    Instead of learning the codebook directly, SimVQ:
    1. Initializes a frozen random codebook
    2. Learns a transformation (neural network) that maps frozen codes to actual codes
    3. Uses the transformed codes for quantization

    Benefits:
    - Simpler training (no complex codebook updates)
    - Can use powerful transformations (MLPs, etc.)
    - Works well with rotation trick
    - No dead code problem

    Args:
        dim: dimension of input/output vectors
        codebook_size: number of codes in the codebook
        codebook_transform: transformation applied to frozen codebook (default: Linear layer)
        init_fn: initialization function for frozen codebook
        channel_first: whether input is channel-first format
        rotation_trick: use rotation trick for gradient estimation
        input_to_quantize_commit_loss_weight: weight for input→quantize commit loss
        commitment_weight: overall weight for commitment loss
        frozen_codebook_dim: dimension of frozen codebook (default: same as dim)
    """
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_transform: Module | None = None,
        init_fn: Callable = identity,
        channel_first = False,
        rotation_trick = True,  # works even better with rotation trick turned on, with no straight through and the commit loss from input to quantize
        input_to_quantize_commit_loss_weight = 0.25,
        commitment_weight = 1.,
        frozen_codebook_dim = None # frozen codebook dim could have different dimensions than projection
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.channel_first = channel_first

        frozen_codebook_dim = default(frozen_codebook_dim, dim)
        codebook = torch.randn(codebook_size, frozen_codebook_dim) * (frozen_codebook_dim ** -0.5)
        codebook = init_fn(codebook)

        # the codebook is actually implicit from a linear layer from frozen gaussian or uniform


        if not exists(codebook_transform):
            codebook_transform = nn.Linear(frozen_codebook_dim, dim, bias = False)

        self.code_transform = codebook_transform

        self.register_buffer('frozen_codebook', codebook)


        # whether to use rotation trick from Fifty et al. 
        # https://arxiv.org/abs/2410.06424

        self.rotation_trick = rotation_trick

        # commit loss weighting - weighing input to quantize a bit less is crucial for it to work

        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight

        # total commitment loss weight

        self.commitment_weight = commitment_weight

    @property
    def codebook(self):
        return self.code_transform(self.frozen_codebook)

    def indices_to_codes(
        self,
        indices
    ):
        implicit_codebook = self.codebook

        frozen_codes = get_at('[c] d, b ... -> b ... d', self.frozen_codebook, indices)
        quantized = self.code_transform(frozen_codes)

        if self.channel_first:
            quantized = rearrange(quantized, 'b ... d -> b d ...')

        return quantized

    def forward(
        self,
        x
    ):
        if self.channel_first:
            x = rearrange(x, 'b d ... -> b ... d')

        x, inverse_pack = pack_one(x, 'b * d')

        implicit_codebook = self.codebook

        with torch.no_grad():
            dist = torch.cdist(x, implicit_codebook)
            indices = dist.argmin(dim = -1)

        # select codes

        quantized = get_at('[c] d, b n -> b n d', implicit_codebook, indices)

        # commit loss and straight through, as was done in the paper

        commit_loss = (
            F.mse_loss(x.detach(), quantized) +
            F.mse_loss(x, quantized.detach()) * self.input_to_quantize_commit_loss_weight
        )

        if self.rotation_trick:
            # rotation trick from @cfifty
            quantized = rotate_to(x, quantized)
        else:
            quantized = (quantized - x).detach() + x

        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices, 'b *')

        if self.channel_first:
            quantized = rearrange(quantized, 'b ... d-> b d ...')

        return quantized, indices, commit_loss * self.commitment_weight

# main

if __name__ == '__main__':

    x = torch.randn(1, 512, 32, 32)

    sim_vq = SimVQ(
        dim = 512,
        codebook_transform = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        ),
        codebook_size = 1024,
        channel_first = True
    )

    quantized, indices, commit_loss = sim_vq(x)

    assert x.shape == quantized.shape
