# src/quantum_hpc/abstract/__init__.py

"""
Abstract Interfaces Package

Base classes and protocols for quantum HPC:
- quantum_processor: Abstract quantum processor interface
- error_correction: Error correction scheme interface
- interconnect: Quantum interconnect definitions

Version: 3.0.0
"""

from .quantum_processor import (
    QuantumProcessor,
    ProcessorType,
    GateType,
    ErrorModel,
    ProcessorCapabilities,
    ProcessorMetrics,
    QuantumState,
    ErrorModelImplementation,
    ProcessorError,
)
from .error_correction import ErrorCorrectionScheme, QECCodeParams
from .interconnect import Interconnect, BasicQuantumInterconnect, LinkConfig

__all__ = [
    # Quantum Processor
    "QuantumProcessor",
    "ProcessorType",
    "GateType",
    "ErrorModel",
    "ProcessorCapabilities",
    "ProcessorMetrics",
    "QuantumState",
    "ErrorModelImplementation",
    "ProcessorError",
    # Error Correction
    "ErrorCorrectionScheme",
    "QECCodeParams",
    # Interconnect
    "Interconnect",
    "BasicQuantumInterconnect",
    "LinkConfig",
]
