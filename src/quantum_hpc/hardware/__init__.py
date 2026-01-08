# src/quantum_hpc/hardware/__init__.py

"""
Hardware Package

Hardware modeling and configuration:
- topology: Qubit connectivity and layout
- noise_model: Quantum noise channels
- calibration: Hardware calibration management

Version: 3.0.0
"""

from .topology import QuantumTopology, QubitPosition, ConnectivityLink
from .noise_model import NoiseModel, NoiseModelConfig, NoiseChannelConfig
from .calibration import CalibrationManager, CalibrationResult

__all__ = [
    # Topology
    "QuantumTopology",
    "QubitPosition",
    "ConnectivityLink",
    # Noise Model
    "NoiseModel",
    "NoiseModelConfig",
    "NoiseChannelConfig",
    # Calibration
    "CalibrationManager",
    "CalibrationResult",
]
