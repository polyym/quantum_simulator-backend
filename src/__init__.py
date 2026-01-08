# src/__init__.py

"""
Quantum Simulator Backend - Source Package

A production-ready quantum computing simulation platform featuring:
- Quantum circuit simulation with QuTiP
- HPC job coordination for distributed simulations
- IonQ-style benchmarking (DRB, application benchmarks)
- Memristor gate acceleration with power metrics
- Surface code error correction

Version: 3.0.0
"""

__version__ = "3.0.0"
__author__ = "polyym"

# Note: Submodules are imported lazily to avoid circular import issues.
# Import them explicitly when needed, e.g.:
#   from src.routers import quantum_system_router
#   from src.services import get_hpc_coordinator

__all__ = [
    "__version__",
]
