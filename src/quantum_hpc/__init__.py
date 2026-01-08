# src/quantum_hpc/__init__.py

"""
Quantum HPC Package

High-performance computing infrastructure for quantum simulations:
- abstract: Base classes for quantum processors and error correction
- devices: QEC implementations (surface code, Bacon-Shor)
- distributed: Job coordination and resource management
- hardware: Topology, noise models, calibration
- virtualization: Simulation engines

Version: 3.0.0
"""

import logging

__version__ = "3.0.0"

logger = logging.getLogger(__name__)

# Subpackages
from . import abstract
from . import devices
from . import distributed
from . import hardware
from . import virtualization

__all__ = [
    "__version__",
    "abstract",
    "devices",
    "distributed",
    "hardware",
    "virtualization",
]
