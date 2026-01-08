# src/routers/__init__.py

"""
API Routers Package

FastAPI routers for all quantum simulation endpoints:
- quantum_system_router: Core quantum operations (create, apply gates, measure)
- hpc_router: HPC job coordination and resource management
- ionq_router: IonQ-style benchmarking (DRB, application benchmarks)
- memristor_router: Memristor gate acceleration
- surface_code_router: Surface code error correction

Version: 3.0.0

Note: Routers are imported lazily to avoid circular import issues.
Import them explicitly when needed, e.g.:
    from src.routers.quantum_system_router import router
"""

__all__ = [
    "quantum_system_router",
    "hpc_router",
    "ionq_router",
    "memristor_router",
    "surface_code_router",
]
