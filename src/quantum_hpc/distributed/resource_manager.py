# src/quantum_hpc/distributed/resource_manager.py

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

@dataclass
class ResourceRequest:
    """
    Describes the resource requirements for a quantum HPC job or sub-task.

    Attributes:
        job_id: Identifier for the requesting job.
        cpu_cores: Number of CPU cores required.
        gpu_cards: Number of GPU cards required.
        memory_gb: Amount of RAM needed in GB.
        duration_estimate: Estimated duration of the job in seconds.
        metadata: Additional arbitrary details (e.g., GPU model, special constraints).
    """
    job_id: str
    cpu_cores: int = 1
    gpu_cards: int = 0
    memory_gb: float = 1.0
    duration_estimate: float = 3600.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResourceAllocation:
    """
    Records an allocation of resources granted to a job or sub-task.

    Attributes:
        job_id: Identifier for the job.
        allocated_cores: Number of CPU cores allocated.
        allocated_gpus: Number of GPU cards allocated.
        allocated_memory: Amount of memory allocated in GB.
        start_time: Time allocation was granted (UNIX timestamp).
        end_time: Optional, if you track when resources should be freed.
        metadata: Additional details about the allocation (e.g. GPU model).
    """
    job_id: str
    allocated_cores: int
    allocated_gpus: int
    allocated_memory: float
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HPCResourceStats:
    """
    Holds a snapshot of current HPC resource usage.
    
    Attributes:
        total_cores: Total CPU cores in the cluster/pool.
        used_cores: CPU cores currently allocated.
        free_cores: CPU cores still available.
        total_gpus: Total GPU cards in the cluster/pool.
        used_gpus: GPU cards currently allocated.
        free_gpus: GPU cards still available.
        total_memory: Total memory (GB) in the cluster/pool.
        used_memory: Memory (GB) allocated.
        free_memory: Memory (GB) still available.
    """
    total_cores: int
    used_cores: int
    free_cores: int
    total_gpus: int
    used_gpus: int
    free_gpus: int
    total_memory: float
    used_memory: float
    free_memory: float

class ResourceManager:
    """
    Simple resource manager for distributed quantum computing tasks.

    In a real HPC environment, you'd integrate with cluster schedulers
    (e.g., Slurm, PBS, Kubernetes) or a library like Dask or Ray. Here,
    we keep a local representation of available resources and track which
    jobs are currently using them.
    """

    def __init__(self,
                 total_cores: int,
                 total_gpus: int,
                 total_memory_gb: float):
        """
        Initialize the resource manager with total available resources.

        Args:
            total_cores: Total CPU cores available in the cluster or node pool.
            total_gpus: Total GPU cards available.
            total_memory_gb: Total RAM available in GB.
        """
        self.total_cores = total_cores
        self.total_gpus = total_gpus
        self.total_memory = total_memory_gb

        self.allocated_cores: int = 0
        self.allocated_gpus: int = 0
        self.allocated_memory: float = 0.0

        # Track active allocations by job_id
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        
        logger.debug(
            f"ResourceManager initialized with total_cores={total_cores}, "
            f"total_gpus={total_gpus}, total_memory_gb={total_memory_gb}"
        )

    def request_resources(self, request: ResourceRequest) -> bool:
        """
        Attempt to allocate resources for a job. If successful, record allocation
        and return True. If insufficient resources, return False.

        Args:
            request: A ResourceRequest describing needed resources.
        """
        try:
            logger.info(
                f"Requesting {request.cpu_cores} CPU, {request.gpu_cards} GPU, "
                f"{request.memory_gb:.2f} GB for job {request.job_id}."
            )

            if self._has_sufficient_resources(request):
                allocation = ResourceAllocation(
                    job_id=request.job_id,
                    allocated_cores=request.cpu_cores,
                    allocated_gpus=request.gpu_cards,
                    allocated_memory=request.memory_gb,
                    metadata=request.metadata
                )
                self._apply_allocation(allocation)
                logger.info(
                    f"Resources allocated for job {request.job_id}. "
                    f"({request.cpu_cores} CPU, {request.gpu_cards} GPU, {request.memory_gb:.2f} GB)"
                )
                return True
            else:
                logger.warning(
                    f"Insufficient resources for job {request.job_id} "
                    f"(Needed: {request.cpu_cores} CPU, {request.gpu_cards} GPU, {request.memory_gb:.2f} GB)."
                )
                return False
        except Exception as e:
            logger.error(f"Error requesting resources for job {request.job_id}: {e}")
            return False

    def release_resources(self, job_id: str) -> bool:
        """
        Free resources allocated to the specified job.

        Args:
            job_id: Identifier for the job whose resources should be released.

        Returns:
            True if resources were successfully released, False if job not found.
        """
        try:
            if job_id not in self.active_allocations:
                logger.warning(f"Cannot release resources; job {job_id} not found.")
                return False
            
            allocation = self.active_allocations.pop(job_id)
            allocation.end_time = time.time()
            
            # Return the resources to the pool
            self.allocated_cores -= allocation.allocated_cores
            self.allocated_gpus -= allocation.allocated_gpus
            self.allocated_memory -= allocation.allocated_memory

            logger.info(
                f"Released resources for job {job_id}. "
                f"({allocation.allocated_cores} CPU, {allocation.allocated_gpus} GPU, "
                f"{allocation.allocated_memory:.2f} GB)"
            )
            return True
        except Exception as e:
            logger.error(f"Error releasing resources for job {job_id}: {e}")
            return False

    def list_allocations(self) -> List[ResourceAllocation]:
        """
        Return a list of active resource allocations.

        Returns:
            A list of ResourceAllocation objects for all currently allocated jobs.
        """
        return list(self.active_allocations.values())

    def get_current_usage(self) -> HPCResourceStats:
        """
        Returns a snapshot of current HPC resource usage (cores, GPUs, memory).
        """
        used_cores = self.allocated_cores
        used_gpus = self.allocated_gpus
        used_mem = self.allocated_memory

        return HPCResourceStats(
            total_cores=self.total_cores,
            used_cores=used_cores,
            free_cores=self.total_cores - used_cores,
            total_gpus=self.total_gpus,
            used_gpus=used_gpus,
            free_gpus=self.total_gpus - used_gpus,
            total_memory=self.total_memory,
            used_memory=used_mem,
            free_memory=self.total_memory - used_mem
        )

    def _has_sufficient_resources(self, request: ResourceRequest) -> bool:
        """
        Check if enough free resources remain for the given request.
        """
        free_cores = self.total_cores - self.allocated_cores
        free_gpus = self.total_gpus - self.allocated_gpus
        free_memory = self.total_memory - self.allocated_memory

        return (
            request.cpu_cores <= free_cores
            and request.gpu_cards <= free_gpus
            and request.memory_gb <= free_memory
        )

    def _apply_allocation(self, allocation: ResourceAllocation) -> None:
        """
        Update internal counters and record an active allocation.
        """
        self.allocated_cores += allocation.allocated_cores
        self.allocated_gpus += allocation.allocated_gpus
        self.allocated_memory += allocation.allocated_memory

        self.active_allocations[allocation.job_id] = allocation
