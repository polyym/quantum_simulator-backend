# src/routers/hpc_router.py

"""
Router exposing HPC coordination endpoints, including job submission,
resource allocation, and job status queries.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Import your HPC modules:
from src.quantum_hpc.distributed.coordinator import HPCJobCoordinator, HPCJob, HPCJobState
from src.quantum_hpc.distributed.resource_manager import ResourceManager, ResourceRequest
from src.quantum_hpc.distributed.resource_manager import HPCResourceStats

logger = logging.getLogger(__name__)
router = APIRouter()

# Example HPC job coordinator and resource manager:
coordinator = HPCJobCoordinator()
resource_manager = ResourceManager(total_cores=128, total_gpus=8, total_memory_gb=256.0)


# -----------------------------
# Pydantic Models
# -----------------------------

class HPCJobCreateRequest(BaseModel):
    """
    Request model for creating and scheduling an HPC job,
    optionally with resource requirements.
    """
    job_id: str
    qubit_count: int
    code_distance: int = 3
    num_cycles: int = 1
    parameters: Dict[str, Any] = {}
    cpu_cores: int = 1
    gpu_cards: int = 0
    memory_gb: float = 1.0

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
        # If you want a callback: callback=my_callback_function
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
