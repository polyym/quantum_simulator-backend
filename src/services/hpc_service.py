# src/services/hpc_service.py

"""
Singleton HPC service providing shared instances of HPCJobCoordinator and ResourceManager.
Ensures all routers use the same resource pool and job tracking.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from src.config import get_settings
from src.quantum_hpc.distributed.coordinator import HPCJobCoordinator, HPCJob, HPCJobState
from src.quantum_hpc.distributed.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

# Module-level lock for thread-safe singleton initialization
# Use RLock (reentrant lock) because get_hpc_coordinator calls get_resource_manager
# while holding the lock, and both functions need to acquire it
_init_lock = threading.RLock()
_hpc_coordinator: Optional["HPCService"] = None
_resource_manager: Optional["ThreadSafeResourceManager"] = None
_quantum_system_manager: Optional["QuantumSystemManager"] = None


class ThreadSafeResourceManager(ResourceManager):
    """
    Thread-safe wrapper around ResourceManager.
    Adds locking to prevent race conditions during resource allocation.
    """

    def __init__(self, total_cores: int, total_gpus: int, total_memory_gb: float):
        super().__init__(total_cores, total_gpus, total_memory_gb)
        self._lock = threading.RLock()

    def request_resources(self, request) -> bool:
        """Thread-safe resource request."""
        with self._lock:
            return super().request_resources(request)

    def release_resources(self, job_id: str) -> bool:
        """Thread-safe resource release."""
        with self._lock:
            return super().release_resources(job_id)

    def get_current_usage(self):
        """Thread-safe usage retrieval."""
        with self._lock:
            return super().get_current_usage()

    def list_allocations(self):
        """Thread-safe allocation listing."""
        with self._lock:
            return super().list_allocations()


class HPCService(HPCJobCoordinator):
    """
    Enhanced HPC Job Coordinator with automatic cleanup of old jobs.
    Provides thread-safe job management with configurable retention.
    """

    def __init__(self, resource_manager: ThreadSafeResourceManager):
        super().__init__()
        self._lock = threading.RLock()
        self._resource_manager = resource_manager
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        """Start background thread for cleaning up old jobs."""
        def cleanup_loop():
            # Wait a bit before first cleanup to ensure initialization is complete
            self._shutdown_event.wait(timeout=60)
            while not self._shutdown_event.is_set():
                try:
                    self._cleanup_old_jobs()
                except Exception as e:
                    logger.error(f"Error in job cleanup: {e}")
                # Check every hour
                self._shutdown_event.wait(timeout=3600)

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.debug("HPC job cleanup thread started")

    def _cleanup_old_jobs(self) -> None:
        """Remove completed/failed/canceled jobs older than retention period."""
        settings = get_settings()
        retention_hours = settings.hpc_job_retention_hours
        cutoff_time = time.time() - (retention_hours * 3600)

        with self._lock:
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if job.state in [HPCJobState.COMPLETED, HPCJobState.FAILED, HPCJobState.CANCELED]:
                    if job.end_time and job.end_time < cutoff_time:
                        jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                # Also release any resources that might still be allocated
                self._resource_manager.release_resources(job_id)

            if jobs_to_remove:
                logger.info(f"Cleaned up {len(jobs_to_remove)} old HPC jobs")

            # Also enforce max jobs limit
            max_jobs = settings.hpc_max_jobs
            if len(self.jobs) > max_jobs:
                # Remove oldest completed jobs first
                completed_jobs = [
                    (jid, j) for jid, j in self.jobs.items()
                    if j.state in [HPCJobState.COMPLETED, HPCJobState.FAILED, HPCJobState.CANCELED]
                ]
                completed_jobs.sort(key=lambda x: x[1].end_time or 0)

                to_remove = len(self.jobs) - max_jobs
                for job_id, _ in completed_jobs[:to_remove]:
                    del self.jobs[job_id]
                    self._resource_manager.release_resources(job_id)

    def submit_job(self, job: HPCJob) -> str:
        """Thread-safe job submission."""
        with self._lock:
            return super().submit_job(job)

    def get_job_status(self, job_id: str) -> Optional[HPCJob]:
        """Thread-safe job status retrieval."""
        with self._lock:
            return super().get_job_status(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Thread-safe job cancellation with resource release."""
        with self._lock:
            result = super().cancel_job(job_id)
            if result:
                self._resource_manager.release_resources(job_id)
            return result

    def list_jobs(self):
        """Thread-safe job listing."""
        with self._lock:
            return super().list_jobs()

    def shutdown(self) -> None:
        """Gracefully shutdown the HPC service."""
        logger.info("Shutting down HPC service...")
        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        logger.info("HPC service shutdown complete")


class QuantumSystemManager:
    """
    Manages quantum system instances with TTL-based cleanup and size limits.
    Thread-safe implementation to prevent memory leaks.
    """

    def __init__(self):
        self._systems: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        """Start background thread for cleaning up expired systems."""
        def cleanup_loop():
            # Wait a bit before first cleanup to ensure initialization is complete
            self._shutdown_event.wait(timeout=30)
            while not self._shutdown_event.is_set():
                try:
                    self._cleanup_expired_systems()
                except Exception as e:
                    logger.error(f"Error in quantum system cleanup: {e}")
                # Check every 5 minutes
                self._shutdown_event.wait(timeout=300)

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.debug("Quantum system cleanup thread started")

    def _cleanup_expired_systems(self) -> None:
        """Remove systems that have exceeded their TTL."""
        settings = get_settings()
        ttl_hours = settings.quantum_system_ttl_hours
        cutoff_time = datetime.now() - timedelta(hours=ttl_hours)

        with self._lock:
            systems_to_remove = []
            for system_id, data in self._systems.items():
                created_at = data.get("created_at")
                last_accessed = data.get("last_accessed", created_at)
                if last_accessed and last_accessed < cutoff_time:
                    systems_to_remove.append(system_id)

            for system_id in systems_to_remove:
                del self._systems[system_id]

            if systems_to_remove:
                logger.info(f"Cleaned up {len(systems_to_remove)} expired quantum systems")

    def create_system(self, system_id: str, engine: Any, description: Optional[str] = None) -> bool:
        """
        Create a new quantum system.

        Args:
            system_id: Unique identifier for the system
            engine: QuantumSimulationEngine instance
            description: Optional description

        Returns:
            True if created, False if already exists or limit reached
        """
        settings = get_settings()

        with self._lock:
            if system_id in self._systems:
                return False

            if len(self._systems) >= settings.quantum_max_systems:
                # Try to clean up expired systems first
                self._cleanup_expired_systems()

                # If still at limit, reject
                if len(self._systems) >= settings.quantum_max_systems:
                    logger.warning(f"Maximum quantum systems limit ({settings.quantum_max_systems}) reached")
                    return False

            self._systems[system_id] = {
                "engine": engine,
                "description": description,
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
            }
            return True

    def get_system(self, system_id: str) -> Optional[Any]:
        """
        Get a quantum system by ID, updating last access time.

        Args:
            system_id: System identifier

        Returns:
            QuantumSimulationEngine or None if not found
        """
        with self._lock:
            if system_id not in self._systems:
                return None

            self._systems[system_id]["last_accessed"] = datetime.now()
            return self._systems[system_id]["engine"]

    def delete_system(self, system_id: str) -> bool:
        """
        Delete a quantum system.

        Args:
            system_id: System identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if system_id in self._systems:
                del self._systems[system_id]
                return True
            return False

    def exists(self, system_id: str) -> bool:
        """Check if a system exists."""
        with self._lock:
            return system_id in self._systems

    def list_systems(self) -> Dict[str, Dict[str, Any]]:
        """List all systems with metadata (excluding engine objects)."""
        with self._lock:
            return {
                sid: {
                    "description": data.get("description"),
                    "created_at": data.get("created_at").isoformat() if data.get("created_at") else None,
                    "last_accessed": data.get("last_accessed").isoformat() if data.get("last_accessed") else None,
                    "num_qubits": data["engine"].num_qubits if data.get("engine") else None,
                }
                for sid, data in self._systems.items()
            }

    def count(self) -> int:
        """Get the number of active systems."""
        with self._lock:
            return len(self._systems)

    def shutdown(self) -> None:
        """Gracefully shutdown the quantum system manager."""
        logger.info("Shutting down quantum system manager...")
        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        with self._lock:
            self._systems.clear()
        logger.info("Quantum system manager shutdown complete")


def get_resource_manager() -> ThreadSafeResourceManager:
    """
    Get the singleton ThreadSafeResourceManager instance.

    Returns:
        Shared ThreadSafeResourceManager instance
    """
    global _resource_manager

    if _resource_manager is None:
        with _init_lock:
            if _resource_manager is None:
                settings = get_settings()
                _resource_manager = ThreadSafeResourceManager(
                    total_cores=settings.hpc_total_cores,
                    total_gpus=settings.hpc_total_gpus,
                    total_memory_gb=settings.hpc_total_memory_gb,
                )
                logger.info(
                    f"Initialized shared ResourceManager: "
                    f"{settings.hpc_total_cores} cores, "
                    f"{settings.hpc_total_gpus} GPUs, "
                    f"{settings.hpc_total_memory_gb}GB memory"
                )

    return _resource_manager


def get_hpc_coordinator() -> HPCService:
    """
    Get the singleton HPCService instance.

    Returns:
        Shared HPCService instance
    """
    global _hpc_coordinator

    if _hpc_coordinator is None:
        with _init_lock:
            if _hpc_coordinator is None:
                resource_manager = get_resource_manager()
                _hpc_coordinator = HPCService(resource_manager)
                logger.info("Initialized shared HPCService")

    return _hpc_coordinator


def get_quantum_system_manager() -> QuantumSystemManager:
    """
    Get the singleton QuantumSystemManager instance.

    Returns:
        Shared QuantumSystemManager instance
    """
    global _quantum_system_manager

    if _quantum_system_manager is None:
        with _init_lock:
            if _quantum_system_manager is None:
                _quantum_system_manager = QuantumSystemManager()
                logger.info("Initialized shared QuantumSystemManager")

    return _quantum_system_manager


def cleanup_services() -> None:
    """
    Cleanup all shared services. Call on application shutdown.
    """
    global _hpc_coordinator, _resource_manager, _quantum_system_manager

    if _hpc_coordinator:
        _hpc_coordinator.shutdown()
        _hpc_coordinator = None

    if _quantum_system_manager:
        _quantum_system_manager.shutdown()
        _quantum_system_manager = None

    _resource_manager = None
    logger.info("All shared services cleaned up")
