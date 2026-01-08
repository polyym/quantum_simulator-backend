# src/quantum_hpc/devices/surface_code/__init__.py

"""
Surface Code Package

Implementation of surface code quantum error correction:
- Stabilizer measurements (X and Z type)
- Syndrome decoding with minimum weight matching
- Multi-round QEC cycles
- Logical operations

Version: 3.0.0
"""

from .decoder import SurfaceCodeDecoder, DecodedSyndromeResult
from .stabilizer import SurfaceCodeStabilizer, StabilizerMeasurementResult
from .surface_code_qec import SurfaceCodeQEC
from .logical_ops import SurfaceCodeLogicalOps

__all__ = [
    # Decoder
    "SurfaceCodeDecoder",
    "DecodedSyndromeResult",
    # Stabilizer
    "SurfaceCodeStabilizer",
    "StabilizerMeasurementResult",
    # QEC
    "SurfaceCodeQEC",
    # Logical Operations
    "SurfaceCodeLogicalOps",
]
