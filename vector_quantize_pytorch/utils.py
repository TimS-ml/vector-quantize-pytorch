"""
Utility classes for vector quantization.

This module provides helper classes for composing quantization modules
with other neural network layers in a sequential manner.
"""

import torch
from torch import nn
from torch.nn import Module, ModuleList

# ===================================
# Quantization Module Imports
# ===================================

from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.residual_vq import ResidualVQ, GroupedResidualVQ
from vector_quantize_pytorch.random_projection_quantizer import RandomProjectionQuantizer
from vector_quantize_pytorch.finite_scalar_quantization import FSQ
from vector_quantize_pytorch.lookup_free_quantization import LFQ
from vector_quantize_pytorch.residual_lfq import ResidualLFQ, GroupedResidualLFQ
from vector_quantize_pytorch.residual_fsq import ResidualFSQ, GroupedResidualFSQ
from vector_quantize_pytorch.latent_quantization import LatentQuantize
from vector_quantize_pytorch.sim_vq import SimVQ
from vector_quantize_pytorch.residual_sim_vq import ResidualSimVQ

# Tuple of all quantization classes for type checking
QUANTIZE_KLASSES = (
    VectorQuantize,
    ResidualVQ,
    GroupedResidualVQ,
    RandomProjectionQuantizer,
    FSQ,
    LFQ,
    SimVQ,
    ResidualSimVQ,
    ResidualLFQ,
    GroupedResidualLFQ,
    ResidualFSQ,
    GroupedResidualFSQ,
    LatentQuantize
)

# ===================================
# Sequential Wrapper for Quantizers
# ===================================

class Sequential(Module):
    """
    Special Sequential module for composing quantizers with other layers.

    This module is designed to work with quantization layers that return
    multiple outputs (quantized tensor, indices, loss, etc.). It ensures:
    - Exactly one quantizer in the sequence
    - Proper handling of quantizer outputs
    - Pass-through of kwargs to the quantizer

    Example:
        Sequential(
            nn.Linear(512, 256),
            VectorQuantize(dim=256, codebook_size=512),
            nn.Linear(256, 512)
        )

    Args:
        *fns: sequence of modules, must contain exactly one quantizer
    """
    def __init__(
        self,
        *fns: Module
    ):
        super().__init__()
        # Verify exactly one quantizer is present
        assert sum([int(isinstance(fn, QUANTIZE_KLASSES)) for fn in fns]) == 1, 'this special Sequential must contain exactly one quantizer'

        self.fns = ModuleList(fns)

    def forward(
        self,
        x,
        **kwargs
    ):
        """
        Forward pass through the sequential layers.

        Non-quantizer layers receive only the tensor.
        The quantizer receives the tensor and all kwargs.

        Returns:
            tuple: (quantized_output, indices, loss, ...) from quantizer
        """
        for fn in self.fns:

            # Regular layers: only pass the tensor
            if not isinstance(fn, QUANTIZE_KLASSES):
                x = fn(x)
                continue

            # Quantizer: pass tensor and kwargs, receive multiple outputs
            x, *rest = fn(x, **kwargs)

        # Return quantizer outputs
        output = (x, *rest)

        return output
