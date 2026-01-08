# src/quantum_hpc/distributed/__init__.py

"""
Distributed HPC Package

Job coordination and resource management:
- coordinator: HPC job lifecycle management
- resource_manager: CPU/GPU/memory allocation
- synchronization: Distributed synchronization primitives

Version: 3.0.0
"""

from .coordinator import HPCJobCoordinator, HPCJob, HPCJobState
from .resource_manager import (
    ResourceManager,
    ResourceRequest,
    ResourceAllocation,
    HPCResourceStats,
)
from .synchronization import DistributedSynchronization, SynchronizationConfig

__all__ = [
    # Coordinator
    "HPCJobCoordinator",
    "HPCJob",
    "HPCJobState",
    # Resource Manager
    "ResourceManager",
    "ResourceRequest",
    "ResourceAllocation",
    "HPCResourceStats",
    # Synchronization
    "DistributedSynchronization",
    "SynchronizationConfig",
]
