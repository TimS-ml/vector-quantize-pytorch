"""
Vector Quantization PyTorch

A comprehensive library of vector quantization methods for neural networks.

Available Quantization Methods:
- VectorQuantize: Standard VQ-VAE with EMA updates or learnable codebook
- FSQ: Finite Scalar Quantization (no learned codebook, per-dimension quantization)
- LFQ: Lookup-Free Quantization (binary quantization with entropy loss)
- SimVQ: Simplified VQ with implicit neural codebook
- LatentQuantize: Latent quantization for disentangled representations
- ResidualVQ/ResidualLFQ/ResidualFSQ/ResidualSimVQ: Residual quantization variants
- GroupedResidualVQ/GroupedResidualLFQ/GroupedResidualFSQ: Grouped residual quantization
- RandomProjectionQuantizer: Random projection-based quantization
- BinaryMapper: Binary mapping for efficient quantization

Utilities:
- Sequential: Special sequential module for composing quantizers with other layers

For more information, see the documentation for each module.
"""

# Core quantization modules
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.residual_vq import ResidualVQ, GroupedResidualVQ
from vector_quantize_pytorch.random_projection_quantizer import RandomProjectionQuantizer
from vector_quantize_pytorch.finite_scalar_quantization import FSQ
from vector_quantize_pytorch.lookup_free_quantization import LFQ
from vector_quantize_pytorch.residual_lfq import ResidualLFQ, GroupedResidualLFQ
from vector_quantize_pytorch.residual_fsq import ResidualFSQ, GroupedResidualFSQ
from vector_quantize_pytorch.latent_quantization import LatentQuantize

# Simplified and specialized quantizers
from vector_quantize_pytorch.sim_vq import SimVQ
from vector_quantize_pytorch.residual_sim_vq import ResidualSimVQ

from vector_quantize_pytorch.binary_mapper import BinaryMapper

# Utility classes
from vector_quantize_pytorch.utils import Sequential
