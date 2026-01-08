# src/quantum_system/__init__.py

"""
Quantum System Package

High-level quantum system abstractions:
- error_correction: Quantum error correction (Steane code, error metrics)
- language: Quantum circuit compilation and instruction sets
- network: Quantum network protocols and switches

Version: 3.0.0
"""

from .error_correction import ErrorCorrection, ErrorMetrics, ErrorType, SteaneCode
from .language import QuantumCompiler, Instruction, LanguageLevel, InstructionSetArchitecture
from .network import QuantumNetwork, QuantumSwitch, NetworkType, TransmissionMode, NetworkMetrics

__all__ = [
    # Error Correction
    "ErrorCorrection",
    "ErrorMetrics",
    "ErrorType",
    "SteaneCode",
    # Language
    "QuantumCompiler",
    "Instruction",
    "LanguageLevel",
    "InstructionSetArchitecture",
    # Network
    "QuantumNetwork",
    "QuantumSwitch",
    "NetworkType",
    "TransmissionMode",
    "NetworkMetrics",
]
