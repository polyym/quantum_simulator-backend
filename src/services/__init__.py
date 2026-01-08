# src/services/__init__.py

"""
Shared services for the Quantum Simulator Backend.
Provides singleton instances of core services like HPC coordination and resource management.
"""

from .hpc_service import (
    get_hpc_coordinator,
    get_resource_manager,
    get_quantum_system_manager,
    cleanup_services,
    HPCService,
    QuantumSystemManager,
)

__all__ = [
    "get_hpc_coordinator",
    "get_resource_manager",
    "get_quantum_system_manager",
    "cleanup_services",
    "HPCService",
    "QuantumSystemManager",
]
