# src/memristor_gates/__init__.py

"""
Memristor Gates Package

Memristor-based quantum gate acceleration:
- Parallel gate execution (up to 4 concurrent operations)
- 4x8 crossbar configuration
- Power metrics (static, dynamic, total energy)

Version: 3.0.0
"""

from .enhanced_gates import (
    EnhancedMemristorCrossbar,
    ParallelQuantumMemristorAccelerator,
    ParallelExecutionUnit,
    PowerMetrics,
    GateType,
)

__all__ = [
    "EnhancedMemristorCrossbar",
    "ParallelQuantumMemristorAccelerator",
    "ParallelExecutionUnit",
    "PowerMetrics",
    "GateType",
]
