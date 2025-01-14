# src/quantum_hpc/__init__.py

"""
Quantum HPC package initialization.
Provides HPC-based quantum simulations, distributed coordination,
hardware models, and virtualization features.
"""

import logging

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# Optionally re-export key modules/classes for convenience:
# from .abstract.quantum_processor import QuantumProcessor
# from .abstract.error_correction import ErrorCorrectionScheme
# from .distributed.coordinator import HPCJobCoordinator
# from .hardware.topology import QuantumTopology
# from .virtualization.simulation import QuantumSimulationEngine

__all__ = [
    # 'QuantumProcessor',
    # 'ErrorCorrectionScheme',
    # 'HPCJobCoordinator',
    # 'QuantumTopology',
    # 'QuantumSimulationEngine'
]
