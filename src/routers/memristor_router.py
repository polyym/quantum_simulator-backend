# src/routers/memristor_router.py

"""
Router exposing memristor-based quantum gate operations and parallel executions,
with optional HPC integration for large or distributed memristor tasks.

==============================================================================
                          ⚠️  EXPERIMENTAL / SPECULATIVE  ⚠️
==============================================================================

IMPORTANT: This module uses THEORETICAL memristor-based quantum gate models.
There is currently NO experimentally validated mechanism for memristors to
implement quantum gate operations while preserving quantum coherence.

See src/memristor_gates/enhanced_gates.py for detailed disclaimers.

For validated quantum simulation, use the /qsystem endpoints instead.
==============================================================================
"""

import logging
import re
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any

# Local usage references:
from src.memristor_gates.enhanced_gates import (
    ParallelQuantumMemristorAccelerator,
    EnhancedMemristorCrossbar,
    PowerMetrics
)

# Shared services
from src.services import get_hpc_coordinator, get_resource_manager
from src.config import get_settings

# HPC types
from src.quantum_hpc.distributed.coordinator import HPCJob
from src.quantum_hpc.distributed.resource_manager import ResourceRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# Memristor accelerator instance with proper cleanup support
_accelerator = None


def get_accelerator() -> ParallelQuantumMemristorAccelerator:
    """Get or create the memristor accelerator singleton."""
    global _accelerator
    if _accelerator is None:
        _accelerator = ParallelQuantumMemristorAccelerator(max_parallel_ops=4)
    return _accelerator


# -------------------------------------------------------------------------
# Pydantic Models with Validation
# -------------------------------------------------------------------------

class MemristorOp(BaseModel):
    gate: str = Field(..., min_length=1, max_length=20)
    qubits: List[int] = Field(..., min_length=1, max_length=10)
    state_dim: int = Field(default=2, ge=2, le=16)

    @field_validator('qubits')
    @classmethod
    def validate_qubits(cls, v):
        settings = get_settings()
        for qubit in v:
            if qubit < 0 or qubit >= settings.quantum_max_qubits:
                raise ValueError(f"Qubit index must be between 0 and {settings.quantum_max_qubits - 1}")
        return v


class MemristorCircuitRequest(BaseModel):
    circuit_id: str = Field(..., min_length=1, max_length=100)
    operations: List[MemristorOp] = Field(..., min_length=1, max_length=10000)

    @field_validator('circuit_id')
    @classmethod
    def validate_circuit_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("circuit_id must contain only alphanumeric characters, underscores, and hyphens")
        return v


class MemristorCircuitResponse(BaseModel):
    circuit_id: str
    final_state: List[float]
    power_metrics: Dict[str, float]
    experimental_warning: str = (
        "EXPERIMENTAL: Results are from a theoretical memristor model with NO "
        "experimental validation. Power metrics are speculative. Do not use for "
        "scientific claims. For validated quantum simulation, use /qsystem endpoints."
    )


@router.post("/run_circuit", response_model=MemristorCircuitResponse)
async def run_memristor_circuit(payload: MemristorCircuitRequest):
    """
    Execute a memristor-based quantum circuit with parallel operations (locally),
    returning final state and partial power metrics.
    """
    try:
        accelerator = get_accelerator()

        # Convert operations to the format expected by the accelerator
        circuit_ops = []
        for op in payload.operations:
            circuit_ops.append({
                'gate': op.gate,
                'qubits': op.qubits,
                'state_dim': op.state_dim
            })

        # Execute the circuit using the memristor accelerator
        state_matrix, metrics = await accelerator.execute_quantum_circuit(circuit_ops)

        # Convert state matrix to a list of floats (real parts for display)
        if state_matrix is not None:
            # Flatten the state matrix and take absolute values for probability-like display
            state_flat = state_matrix.flatten()
            final_state = [float(np.abs(x)) for x in state_flat[:min(len(state_flat), 16)]]
        else:
            # Default state if no operations were executed
            final_state = [1.0, 0.0]

        # Calculate additional power metrics using the crossbar
        total_power_metrics = PowerMetrics()
        crossbar = EnhancedMemristorCrossbar(rows=4, cols=8)
        for op in circuit_ops:
            op_metrics = crossbar.calculate_power_metrics(op)
            total_power_metrics.total_energy += op_metrics.total_energy
            total_power_metrics.dynamic_power += op_metrics.dynamic_power
            total_power_metrics.static_power += op_metrics.static_power

        # Combine accelerator metrics with detailed crossbar metrics
        power_metrics = {
            "total_energy_nJ": metrics.get("total_energy_nJ", 0) + total_power_metrics.total_energy * 1e9,
            "dynamic_power_mW": metrics.get("accumulated_dynamic_power_mW", 0) + total_power_metrics.dynamic_power * 1e3,
            "static_power_mW": total_power_metrics.static_power * 1e3,
            "num_operations": len(circuit_ops)
        }

        logger.info(f"Memristor circuit executed: id={payload.circuit_id}, ops={len(payload.operations)}")
    except Exception as e:
        logger.error(f"Memristor circuit execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit execution failed: {str(e)}")

    return MemristorCircuitResponse(
        circuit_id=payload.circuit_id,
        final_state=final_state,
        power_metrics=power_metrics
    )


# -------------------------------------------------------------------------
# HPC Synergy Endpoint
# -------------------------------------------------------------------------

class HPCMemristorRequest(BaseModel):
    """
    Request model for distributing a large memristor circuit via HPC.
    We include HPC resource fields and the circuit data (operations).
    """
    job_id: str = Field(..., min_length=1, max_length=100)
    circuit_id: str = Field(..., min_length=1, max_length=100)
    operations: List[MemristorOp] = Field(..., min_length=1, max_length=10000)
    # HPC resource requirements
    cpu_cores: int = Field(default=1, ge=1, le=256)
    gpu_cards: int = Field(default=0, ge=0, le=16)
    memory_gb: float = Field(default=1.0, gt=0, le=1024)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('job_id', 'circuit_id')
    @classmethod
    def validate_ids(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("ID must contain only alphanumeric characters, underscores, and hyphens")
        return v


class HPCMemristorResponse(BaseModel):
    """
    Response after HPC submission for a memristor circuit.
    """
    job_id: str
    message: str
    experimental_warning: str = (
        "EXPERIMENTAL: Memristor-based quantum gates are theoretical with NO "
        "experimental validation. For validated simulations, use /qsystem endpoints."
    )


@router.post("/submit_hpc_memristor", response_model=HPCMemristorResponse)
def submit_hpc_memristor(payload: HPCMemristorRequest):
    """
    Submit a large memristor circuit to HPC, allocating resources and scheduling an HPC job.
    The HPC job can handle many parallel gates, advanced power modeling, or multi-node tasks.
    """
    coordinator = get_hpc_coordinator()
    resource_manager = get_resource_manager()

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
        "operations": [op.model_dump() for op in payload.operations],
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
