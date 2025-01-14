# src/routers/quantum_system_router.py

"""
Router exposing quantum system operations, including circuit creation,
applying gates, and measurement. Optionally integrates with HPC for
large or distributed quantum simulations.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Example: integrate with HPC modules
from src.quantum_hpc.distributed.coordinator import HPCJobCoordinator, HPCJob
from src.quantum_hpc.distributed.resource_manager import ResourceManager, ResourceRequest
from src.quantum_hpc.distributed.resource_manager import HPCResourceStats
from src.quantum_hpc.distributed.coordinator import HPCJobState

# from src.quantum_system.error_correction import ErrorCorrection
# from src.quantum_system.language import QuantumCompiler
# from src.quantum_system.network import QuantumNetwork
# from src.quantum_hpc.virtualization.simulation import QuantumSimulationEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store of active system simulations (for local usage)
QUANTUM_SYSTEMS: Dict[str, Any] = {}

# Optional HPC synergy: if you want HPC distribution for large tasks
coordinator = HPCJobCoordinator()
resource_manager = ResourceManager(total_cores=64, total_gpus=4, total_memory_gb=128.0)

# -------------------------------------------------------------------------
# Pydantic models for local usage
# -------------------------------------------------------------------------
class CreateSystemRequest(BaseModel):
    system_id: str
    num_qubits: int
    description: Optional[str] = None

class CreateSystemResponse(BaseModel):
    system_id: str
    message: str

class ApplyOperationRequest(BaseModel):
    system_id: str
    operation: str
    qubits: List[int]
    params: Dict[str, Any] = {}

class ApplyOperationResponse(BaseModel):
    message: str

class MeasureRequest(BaseModel):
    system_id: str
    qubits: List[int]

class MeasureResponse(BaseModel):
    outcomes: Dict[str, float]

# -------------------------------------------------------------------------
# Local Endpoints (Small-Scale or Non-HPC)
# -------------------------------------------------------------------------

@router.post("/create_system", response_model=CreateSystemResponse)
def create_system(payload: CreateSystemRequest):
    """
    Create a new local quantum system instance (e.g., a small simulation).
    For HPC synergy, see /submit_distributed_simulation.
    """
    if payload.system_id in QUANTUM_SYSTEMS:
        raise HTTPException(status_code=400, detail="System ID already exists.")
    # Initialize or integrate with your quantum engine:
    # engine = QuantumSimulationEngine(num_qubits=payload.num_qubits)
    engine = {
        "num_qubits": payload.num_qubits,
        "description": payload.description
    }  # mock

    QUANTUM_SYSTEMS[payload.system_id] = engine

    return CreateSystemResponse(
        system_id=payload.system_id,
        message=f"Quantum system '{payload.system_id}' created with {payload.num_qubits} qubits."
    )

@router.post("/apply_operation", response_model=ApplyOperationResponse)
def apply_operation(payload: ApplyOperationRequest):
    """
    Apply a quantum operation (gate or error correction step) to a local system.
    """
    system_id = payload.system_id
    if system_id not in QUANTUM_SYSTEMS:
        raise HTTPException(status_code=404, detail="System not found.")

    # engine = QUANTUM_SYSTEMS[system_id]
    # engine.apply_gate(payload.operation, payload.qubits, payload.params)

    return ApplyOperationResponse(
        message=(
            f"Operation '{payload.operation}' applied on system '{system_id}' "
            f"for qubits {payload.qubits}."
        )
    )

@router.post("/measure", response_model=MeasureResponse)
def measure(payload: MeasureRequest):
    """
    Measure specified qubits in the local system and return outcome probabilities.
    """
    system_id = payload.system_id
    if system_id not in QUANTUM_SYSTEMS:
        raise HTTPException(status_code=404, detail="System not found.")

    # engine = QUANTUM_SYSTEMS[system_id]
    # results = engine.measure_probabilities(payload.qubits)
    # For demonstration:
    results = {str(q): 0.5 for q in payload.qubits}

    return MeasureResponse(outcomes=results)

@router.delete("/delete_system")
def delete_system(system_id: str):
    """
    Delete a local quantum system from memory.
    """
    if system_id in QUANTUM_SYSTEMS:
        del QUANTUM_SYSTEMS[system_id]
        return {"message": f"Quantum system '{system_id}' deleted."}
    raise HTTPException(status_code=404, detail="System not found.")

# -------------------------------------------------------------------------
# HPC Synergy: Distributed Simulation Endpoint
# -------------------------------------------------------------------------

class DistributedSimRequest(BaseModel):
    """
    Request model for distributing a quantum simulation task via HPC.
    
    Attributes:
        job_id: HPC job identifier.
        num_qubits: Number of qubits for the simulation.
        parameters: Additional simulation or HPC configuration details.
        cpu_cores, gpu_cards, memory_gb: HPC resource requests.
        code_distance, num_cycles: If relevant for QEC or HPC tasks.
    """
    job_id: str
    num_qubits: int
    code_distance: int = 3
    num_cycles: int = 1
    cpu_cores: int = 1
    gpu_cards: int = 0
    memory_gb: float = 1.0
    parameters: Dict[str, Any] = {}

class DistributedSimResponse(BaseModel):
    """
    Response after HPC job is submitted for a distributed quantum simulation.
    """
    job_id: str
    message: str

@router.post("/submit_distributed_simulation", response_model=DistributedSimResponse)
def submit_distributed_simulation(payload: DistributedSimRequest):
    """
    Submit a large or distributed quantum simulation as an HPC job.
    Allocates HPC resources, then uses HPCJobCoordinator to queue the job.
    The HPC job can run big circuits, multi-cycle QEC, or IonQ-like tasks
    in a distributed environment.

    Usage:
      - Provide 'job_id' for HPC tracking
      - Resource params (cpu_cores, gpu_cards, memory_gb)
      - 'parameters' can hold circuit data, gate lists, QEC configs, etc.
    """
    # 1) HPC resource allocation
    req = ResourceRequest(
        job_id=payload.job_id,
        cpu_cores=payload.cpu_cores,
        gpu_cards=payload.gpu_cards,
        memory_gb=payload.memory_gb,
        duration_estimate=3600.0,  # or something from 'parameters'
        metadata=payload.parameters
    )
    allocated = resource_manager.request_resources(req)
    if not allocated:
        raise HTTPException(status_code=400, detail="Insufficient HPC resources for this simulation.")

    # 2) HPC job creation
    job = HPCJob(
        job_id=payload.job_id,
        qubit_count=payload.num_qubits,
        code_distance=payload.code_distance,
        num_cycles=payload.num_cycles,
        parameters=payload.parameters
    )
    try:
        coordinator.submit_job(job)
    except ValueError as e:
        # If job_id conflict or other HPC error, release resources
        resource_manager.release_resources(payload.job_id)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"Distributed simulation submitted: job_id={payload.job_id}, "
        f"qubits={payload.num_qubits}, HPC resources allocated."
    )
    return DistributedSimResponse(
        job_id=payload.job_id,
        message=(f"Distributed simulation submitted. HPC job '{payload.job_id}' is queued. "
                 "Use /hpc/job_status to monitor progress.")
    )
