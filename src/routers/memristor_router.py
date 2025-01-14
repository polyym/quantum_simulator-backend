# src/routers/memristor_router.py

"""
Router exposing memristor-based quantum gate operations and parallel executions,
with optional HPC integration for large or distributed memristor tasks.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Tuple

# Local usage references:
# from src.memristor_gates.enhanced_gates import (
#     ParallelQuantumMemristorAccelerator,
#     EnhancedMemristorCrossbar
# )

# HPC synergy references:
from src.quantum_hpc.distributed.coordinator import HPCJobCoordinator, HPCJob
from src.quantum_hpc.distributed.resource_manager import ResourceManager, ResourceRequest
from src.quantum_hpc.distributed.coordinator import HPCJobState

logger = logging.getLogger(__name__)
router = APIRouter()

# If you want HPC synergy, instantiate HPC modules (if not done globally):
coordinator = HPCJobCoordinator()
resource_manager = ResourceManager(total_cores=64, total_gpus=4, total_memory_gb=128.0)

# -------------------------------------------------------------------------
# Local Memristor Endpoint
# -------------------------------------------------------------------------

class MemristorOp(BaseModel):
    gate: str
    qubits: List[int]
    state_dim: int = 2

class MemristorCircuitRequest(BaseModel):
    circuit_id: str
    operations: List[MemristorOp]

class MemristorCircuitResponse(BaseModel):
    circuit_id: str
    final_state: List[float]
    power_metrics: Dict[str, float]

@router.post("/run_circuit", response_model=MemristorCircuitResponse)
async def run_memristor_circuit(payload: MemristorCircuitRequest):
    """
    Execute a memristor-based quantum circuit with parallel operations (locally),
    returning final state and partial power metrics.
    """
    # If you had a real accelerator instance:
    # final_state, metrics = await accelerator.execute_quantum_circuit(payload.operations)
    # For demonstration:
    final_state = [1.0, 0.0]  # trivial
    metrics = {"total_energy_pj": 0.1, "avg_dynamic_power_pw": 0.01}

    logger.debug(f"Running local memristor circuit: id={payload.circuit_id}, ops={len(payload.operations)}")
    return MemristorCircuitResponse(
        circuit_id=payload.circuit_id,
        final_state=final_state,
        power_metrics=metrics
    )

# -------------------------------------------------------------------------
# HPC Synergy Endpoint (Optional)
# -------------------------------------------------------------------------

class HPCMemristorRequest(BaseModel):
    """
    Request model for distributing a large memristor circuit via HPC.
    We include HPC resource fields and the circuit data (operations).
    """
    job_id: str
    circuit_id: str
    operations: List[MemristorOp]
    # HPC resource requirements
    cpu_cores: int = 1
    gpu_cards: int = 0
    memory_gb: float = 1.0
    parameters: Dict[str, Any] = {}

class HPCMemristorResponse(BaseModel):
    """
    Response after HPC submission for a memristor circuit.
    """
    job_id: str
    message: str

@router.post("/submit_hpc_memristor", response_model=HPCMemristorResponse)
def submit_hpc_memristor(payload: HPCMemristorRequest):
    """
    Submit a large memristor circuit to HPC, allocating resources and scheduling an HPC job.
    The HPC job can handle many parallel gates, advanced power modeling, or multi-node tasks.
    """
    # 1) HPC resource allocation
    request = ResourceRequest(
        job_id=payload.job_id,
        cpu_cores=payload.cpu_cores,
        gpu_cards=payload.gpu_cards,
        memory_gb=payload.memory_gb,
        duration_estimate=3600.0,  # Could be from len(operations) or payload.parameters
        metadata=payload.parameters
    )
    allocated = resource_manager.request_resources(request)
    if not allocated:
        raise HTTPException(status_code=400, detail="Insufficient HPC resources for memristor circuit.")

    # 2) HPC job creation
    # We store the circuit data in the HPCJob 'parameters' so that coordinator logic can retrieve it.
    job_params = {
        "circuit_id": payload.circuit_id,
        "operations": [op.dict() for op in payload.operations],
        **payload.parameters
    }
    job = HPCJob(
        job_id=payload.job_id,
        qubit_count=32,  # or deduce from max qubit index in operations
        code_distance=1,  # Not used for memristors typically
        num_cycles=1,  # You might interpret this differently
        parameters=job_params
    )

    try:
        coordinator.submit_job(job)
    except ValueError as e:
        # If job_id conflict or HPC error, release resources
        resource_manager.release_resources(payload.job_id)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"Memristor HPC job submitted: job_id={payload.job_id}, circuit_id={payload.circuit_id}, "
        f"ops={len(payload.operations)} HPC resources allocated."
    )
    return HPCMemristorResponse(
        job_id=payload.job_id,
        message=(
            f"HPC memristor job '{payload.job_id}' submitted successfully. "
            "Use /hpc/job_status to monitor progress."
        )
    )
