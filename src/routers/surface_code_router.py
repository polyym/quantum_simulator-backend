# src/routers/surface_code_router.py

"""
Router exposing surface code operations (stabilizer measurement, decoding, QEC cycles),
including an optional multi-round HPC-based route for large distances or repeated cycles.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Example references to HPC modules (if you want HPC synergy for multi-round QEC)
from src.quantum_hpc.distributed.coordinator import HPCJobCoordinator, HPCJob
from src.quantum_hpc.distributed.resource_manager import ResourceManager, ResourceRequest
from src.quantum_hpc.distributed.resource_manager import HPCResourceStats
from src.quantum_hpc.distributed.coordinator import HPCJobState

# Example references to surface code classes
# from src.quantum_hpc.devices.surface_code.decoder import SurfaceCodeDecoder
# from src.quantum_hpc.devices.surface_code.stabilizer import SurfaceCodeStabilizer

logger = logging.getLogger(__name__)
router = APIRouter()

# If HPC synergy is desired, create global references:
coordinator = HPCJobCoordinator()
resource_manager = ResourceManager(total_cores=64, total_gpus=4, total_memory_gb=128.0)

# ---------------------------------------------------
# Pydantic Models for existing single-round endpoints
# ---------------------------------------------------

class StabilizerRequest(BaseModel):
    distance: int
    cycle_index: Optional[int] = None

class StabilizerResponse(BaseModel):
    X_stabilizers: List[List[int]]
    Z_stabilizers: List[List[int]]
    metadata: Dict[str, Any]

class DecodeRequest(BaseModel):
    distance: int
    stabilizer_data: Dict[str, Any]

class DecodeResponse(BaseModel):
    corrections: Dict[str, Any]
    logical_errors_detected: List[str]

# ---------------------------------------------------
# Existing Single-Round Endpoints
# ---------------------------------------------------

@router.post("/measure_stabilizers", response_model=StabilizerResponse)
async def measure_stabilizers(payload: StabilizerRequest):
    """
    Measure X- and Z-type stabilizers for a given surface code distance,
    returning raw data and metadata.
    """
    # measurement_result = await stabilizer.measure_all_stabilizers(...)
    # For demonstration, return a mock response
    logger.debug(f"Measuring stabilizers at distance={payload.distance}, cycle={payload.cycle_index}")
    return StabilizerResponse(
        X_stabilizers=[[0, 0], [0, 0]],
        Z_stabilizers=[[0, 0], [0, 0]],
        metadata={"cycle_index": payload.cycle_index, "distance": payload.distance}
    )

@router.post("/decode_syndrome", response_model=DecodeResponse)
async def decode_syndrome(payload: DecodeRequest):
    """
    Decode a measured syndrome from surface code stabilizers, returning
    corrective operations and any detected logical errors.
    """
    # decode_result = decoder.decode_syndrome(...)
    # Mock
    logger.debug(f"Decoding syndrome at distance={payload.distance}, data={payload.stabilizer_data}")
    return DecodeResponse(
        corrections={"qubit_ops": []},
        logical_errors_detected=[]
    )

# ---------------------------------------------------
# Multi-Round QEC Endpoint (Optional HPC Synergy)
# ---------------------------------------------------

class MultiRoundQECRequest(BaseModel):
    """
    Request model for running multiple surface code QEC cycles,
    optionally using HPC resources if you want to scale large distances.
    """
    job_id: str
    distance: int
    rounds: int = 10
    # HPC resource requirements:
    cpu_cores: int = 1
    gpu_cards: int = 0
    memory_gb: float = 1.0
    parameters: Dict[str, Any] = {}

class MultiRoundQECResponse(BaseModel):
    """
    Response for multi-round QEC submission:
      - job_id to track HPC job (if HPC synergy is used).
      - message for immediate feedback.
    """
    job_id: str
    message: str

@router.post("/run_multi_round_qec", response_model=MultiRoundQECResponse)
def run_multi_round_qec(payload: MultiRoundQECRequest):
    """
    Submit a multi-round surface code QEC task. If HPC synergy is enabled,
    we will:
      1) Attempt to allocate resources via ResourceManager.
      2) Create an HPCJob to run multi-round QEC in HPCJobCoordinator.
      3) Return job_id for front-end to track via /hpc/job_status.

    If you prefer local immediate execution, skip HPC synergy or
    directly run multi-round loops here.
    """
    # 1) HPC resource allocation:
    request = ResourceRequest(
        job_id=payload.job_id,
        cpu_cores=payload.cpu_cores,
        gpu_cards=payload.gpu_cards,
        memory_gb=payload.memory_gb,
        duration_estimate=3600.0,  # or something derived from 'rounds' + 'distance'
        metadata=payload.parameters
    )
    allocated = resource_manager.request_resources(request)
    if not allocated:
        raise HTTPException(
            status_code=400,
            detail="Insufficient HPC resources for multi-round QEC task."
        )

    # 2) HPC job creation:
    job = HPCJob(
        job_id=payload.job_id,
        qubit_count=payload.distance**2,  # Just a naive guess if we want # of qubits ~ distance^2
        code_distance=payload.distance,
        num_cycles=payload.rounds,
        parameters=payload.parameters
    )
    try:
        # HPC job coordinator usage
        coordinator.submit_job(job)
    except ValueError as e:
        # If job_id conflict or other error, release HPC resources
        resource_manager.release_resources(payload.job_id)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"Multi-round QEC job submitted: job_id={payload.job_id}, "
        f"distance={payload.distance}, rounds={payload.rounds}"
    )
    return MultiRoundQECResponse(
        job_id=payload.job_id,
        message=(
            f"Multi-round QEC task submitted successfully. HPC job '{payload.job_id}' queued. "
            "Use /hpc/job_status to monitor progress."
        )
    )
