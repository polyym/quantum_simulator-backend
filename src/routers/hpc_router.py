# src/routers/hpc_router.py

"""
Router exposing HPC coordination endpoints, including job submission,
resource allocation, and job status queries.
"""

import logging
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List

# Shared services
from src.services import get_hpc_coordinator, get_resource_manager
from src.config import get_settings

# Import HPC types
from src.quantum_hpc.distributed.coordinator import HPCJob
from src.quantum_hpc.distributed.resource_manager import ResourceRequest

logger = logging.getLogger(__name__)
router = APIRouter()


# -----------------------------
# Pydantic Models with Validation
# -----------------------------

class HPCJobCreateRequest(BaseModel):
    """
    Request model for creating and scheduling an HPC job,
    optionally with resource requirements.
    """
    job_id: str = Field(..., min_length=1, max_length=100)
    qubit_count: int = Field(..., ge=1)
    code_distance: int = Field(default=3, ge=1, le=25)
    num_cycles: int = Field(default=1, ge=1, le=10000)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    cpu_cores: int = Field(default=1, ge=1, le=256)
    gpu_cards: int = Field(default=0, ge=0, le=16)
    memory_gb: float = Field(default=1.0, gt=0, le=1024)

    @field_validator('job_id')
    @classmethod
    def validate_job_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("job_id must contain only alphanumeric characters, underscores, and hyphens")
        return v

    @field_validator('qubit_count')
    @classmethod
    def validate_qubit_count(cls, v):
        settings = get_settings()
        if v > settings.quantum_max_qubits:
            raise ValueError(f"qubit_count must be <= {settings.quantum_max_qubits}")
        return v


class HPCJobCreateResponse(BaseModel):
    """
    Basic response model when a job is submitted.
    """
    job_id: str
    message: str


class HPCJobStatusResponse(BaseModel):
    """
    Reflects current HPC job status in the HPCJobCoordinator.
    """
    job_id: str
    state: str
    start_time: Optional[float]
    end_time: Optional[float]
    progress: float
    result: Dict[str, Any]
    error: Optional[str] = None


class HPCJobListResponse(BaseModel):
    """
    For listing all HPC jobs in the system.
    """
    jobs: List[HPCJobStatusResponse]


class HPCResourceUsageResponse(BaseModel):
    """
    Shows a snapshot of HPC resource usage from ResourceManager.
    """
    total_cores: int
    used_cores: int
    free_cores: int
    total_gpus: int
    used_gpus: int
    free_gpus: int
    total_memory_gb: float
    used_memory_gb: float
    free_memory_gb: float


# -----------------------------
# Routes
# -----------------------------

@router.post("/submit_job", response_model=HPCJobCreateResponse)
def submit_job(payload: HPCJobCreateRequest):
    """
    Submit a new HPC quantum job, optionally requesting resources
    and scheduling via HPCJobCoordinator.
    """
    coordinator = get_hpc_coordinator()
    resource_manager = get_resource_manager()

    # 1) Attempt to allocate resources if desired:
    request = ResourceRequest(
        job_id=payload.job_id,
        cpu_cores=payload.cpu_cores,
        gpu_cards=payload.gpu_cards,
        memory_gb=payload.memory_gb,
        duration_estimate=3600.0,  # or from payload.parameters if relevant
        metadata=payload.parameters
    )
    allocated = resource_manager.request_resources(request)
    if not allocated:
        raise HTTPException(status_code=400, detail="Insufficient HPC resources for this job.")

    # 2) Create HPCJob and submit to HPCJobCoordinator
    job = HPCJob(
        job_id=payload.job_id,
        qubit_count=payload.qubit_count,
        code_distance=payload.code_distance,
        num_cycles=payload.num_cycles,
        parameters=payload.parameters,
    )
    try:
        coordinator.submit_job(job)
    except ValueError as e:
        # If job_id already exists or other issue
        # Release resources to avoid resource leak
        resource_manager.release_resources(payload.job_id)
        raise HTTPException(status_code=400, detail=str(e))

    return HPCJobCreateResponse(
        job_id=payload.job_id,
        message=f"Job '{payload.job_id}' submitted successfully and resources allocated."
    )


@router.get("/job_status", response_model=HPCJobStatusResponse)
def job_status(job_id: str):
    """
    Retrieve the status of an HPC quantum job from HPCJobCoordinator.
    """
    coordinator = get_hpc_coordinator()
    job = coordinator.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return HPCJobStatusResponse(
        job_id=job.job_id,
        state=job.state.value,
        start_time=job.start_time,
        end_time=job.end_time,
        progress=job.progress,
        result=job.result,
        error=job.error
    )


@router.get("/list_jobs", response_model=HPCJobListResponse)
def list_jobs():
    """
    List all HPC jobs known to the coordinator.
    """
    coordinator = get_hpc_coordinator()
    jobs = coordinator.list_jobs()
    response_jobs = []
    for j in jobs:
        response_jobs.append(HPCJobStatusResponse(
            job_id=j.job_id,
            state=j.state.value,
            start_time=j.start_time,
            end_time=j.end_time,
            progress=j.progress,
            result=j.result,
            error=j.error
        ))
    return HPCJobListResponse(jobs=response_jobs)


@router.delete("/cancel_job")
def cancel_job(job_id: str):
    """
    Attempt to cancel an HPC job, releasing resources if allocated.
    """
    coordinator = get_hpc_coordinator()
    resource_manager = get_resource_manager()

    # 1) Release HPC resources first (if allocated).
    resource_released = resource_manager.release_resources(job_id)
    # 2) Cancel job in coordinator.
    canceled = coordinator.cancel_job(job_id)

    if canceled:
        return {"status": "success", "message": f"Job '{job_id}' canceled. Resources released={resource_released}"}
    else:
        return {"status": "warning", "message": f"Unable to cancel job '{job_id}'. Job may not exist or is completed."}


@router.post("/process_queue")
def process_queue():
    """
    Explicitly process HPC queue (placeholder if using manual HPC loops).
    If your HPC tasks run asynchronously, you could do more advanced queue logic here.
    """
    # If you had a real queue, you'd do coordinator.process_queue() or similar.
    return {"message": "HPC queue processed (no-op)."}


@router.get("/resources", response_model=HPCResourceUsageResponse)
def get_hpc_resources():
    """
    Return a snapshot of HPC resource usage from ResourceManager.
    """
    resource_manager = get_resource_manager()
    stats = resource_manager.get_current_usage()
    return HPCResourceUsageResponse(
        total_cores=stats.total_cores,
        used_cores=stats.used_cores,
        free_cores=stats.free_cores,
        total_gpus=stats.total_gpus,
        used_gpus=stats.used_gpus,
        free_gpus=stats.free_gpus,
        total_memory_gb=stats.total_memory,
        used_memory_gb=stats.used_memory,
        free_memory_gb=stats.free_memory
    )
