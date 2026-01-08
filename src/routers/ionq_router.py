# src/routers/ionq_router.py

"""
Router exposing IonQ-like benchmarking endpoints, such as DRB,
application benchmarks, and error mitigation routines, with optional HPC integration.
"""

import logging
import re
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List

# IonQ modules
from src.ionq_benchmarking.core import IonQDevice, ApplicationBenchmarks
from src.ionq_benchmarking.error_mitigation import ErrorMitigation

# Shared services
from src.services import get_hpc_coordinator, get_resource_manager
from src.config import get_settings

# HPC types
from src.quantum_hpc.distributed.coordinator import HPCJob
from src.quantum_hpc.distributed.resource_manager import ResourceRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# IonQ device and benchmark instances
_ionq_devices: Dict[int, IonQDevice] = {}


def get_ionq_device(num_qubits: int) -> IonQDevice:
    """Get or create an IonQ device with the specified number of qubits."""
    if num_qubits not in _ionq_devices:
        _ionq_devices[num_qubits] = IonQDevice(num_qubits=num_qubits)
    return _ionq_devices[num_qubits]


# Error mitigation instance
mitigation = ErrorMitigation(num_variants=25)


# -----------------------------------------------------------
# Pydantic Models for Local IonQ Endpoints with Validation
# -----------------------------------------------------------

class IonQDRBRequest(BaseModel):
    qubits: List[int] = Field(..., min_length=1, max_length=30)
    depth: int = Field(..., ge=1, le=10000)
    p2q: float = Field(default=0.25, ge=0, le=1)

    @field_validator('qubits')
    @classmethod
    def validate_qubits(cls, v):
        settings = get_settings()
        for qubit in v:
            if qubit < 0 or qubit >= settings.quantum_max_qubits:
                raise ValueError(f"Qubit index must be between 0 and {settings.quantum_max_qubits - 1}")
        return v


class IonQDRBResponse(BaseModel):
    success_probability: float


class IonQApplicationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    width: int = Field(..., ge=1)

    @field_validator('width')
    @classmethod
    def validate_width(cls, v):
        settings = get_settings()
        if v > settings.quantum_max_qubits:
            raise ValueError(f"width must be <= {settings.quantum_max_qubits}")
        return v


class IonQApplicationResponse(BaseModel):
    fidelity: float


class IonQErrorMitigationRequest(BaseModel):
    circuit: List[Dict[str, Any]] = Field(..., min_length=1, max_length=10000)
    num_qubits: int = Field(..., ge=1)

    @field_validator('num_qubits')
    @classmethod
    def validate_num_qubits(cls, v):
        settings = get_settings()
        if v > settings.quantum_max_qubits:
            raise ValueError(f"num_qubits must be <= {settings.quantum_max_qubits}")
        return v


class IonQErrorMitigationResponse(BaseModel):
    aggregated_distribution: Dict[str, float]


# -----------------------------------------------------------
# Local IonQ Endpoints (No HPC)
# -----------------------------------------------------------

@router.post("/drb", response_model=IonQDRBResponse)
def run_drb(payload: IonQDRBRequest):
    """
    Endpoint to run Direct Randomized Benchmarking on an IonQ-like device.
    """
    try:
        # Get or create device with enough qubits
        num_qubits = max(payload.qubits) + 1 if payload.qubits else 2
        device = get_ionq_device(num_qubits)

        # Run DRB - returns a dictionary with results
        drb_result = device.run_drb(payload.qubits, payload.depth, payload.p2q)
        success_prob = drb_result.get('survival_probability', 0.0)
        logger.info(f"DRB completed: qubits={payload.qubits}, depth={payload.depth}, p2q={payload.p2q}, success={success_prob:.4f}")
    except Exception as e:
        logger.error(f"DRB failed: {e}")
        raise HTTPException(status_code=500, detail=f"DRB execution failed: {str(e)}")

    return IonQDRBResponse(success_probability=success_prob)


@router.post("/application", response_model=IonQApplicationResponse)
def run_application_benchmark(payload: IonQApplicationRequest):
    """
    Runs an IonQ application benchmark, e.g., Hamiltonian sim or QFT, locally.
    """
    try:
        # Get or create device with the specified width
        device = get_ionq_device(payload.width)
        app_bench = ApplicationBenchmarks(device)

        # Run the benchmark
        fidelity = app_bench.run_benchmark(payload.name, payload.width)
        logger.info(f"Application benchmark completed: name={payload.name}, width={payload.width}, fidelity={fidelity:.4f}")
    except Exception as e:
        logger.error(f"Application benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Application benchmark failed: {str(e)}")

    return IonQApplicationResponse(fidelity=fidelity)


@router.post("/error_mitigation", response_model=IonQErrorMitigationResponse)
def run_error_mitigation(payload: IonQErrorMitigationRequest):
    """
    Applies IonQ-inspired error mitigation (circuit variants + aggregation) locally.
    """
    try:
        # Generate circuit variants
        variants = mitigation.generate_circuit_variants(payload.circuit, payload.num_qubits)

        # Simulate each variant and collect results
        # For now, we generate synthetic measurement distributions based on circuit analysis
        variant_results = []
        for variant in variants:
            # Simulate the variant circuit and get measurement counts
            # This is a simplified simulation - in production, you'd run on actual hardware or a full simulator
            counts = _simulate_circuit_variant(variant, payload.num_qubits)
            variant_results.append(counts)

        # Aggregate results using plurality voting
        aggregated = mitigation.aggregate_results(variant_results)
        logger.info(f"Error mitigation completed: {payload.num_qubits} qubits, {len(variants)} variants")
    except Exception as e:
        logger.error(f"Error mitigation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error mitigation failed: {str(e)}")

    return IonQErrorMitigationResponse(aggregated_distribution=aggregated)


def _simulate_circuit_variant(circuit: List[Dict[str, Any]], num_qubits: int) -> Dict[str, int]:
    """
    Simulate a circuit variant and return measurement counts.
    This is a simplified simulation for demonstration purposes.
    """
    # Generate a realistic-looking distribution based on circuit structure
    num_shots = 100
    counts = {}

    # Count the number of entangling gates to estimate distribution complexity
    num_entangling = sum(1 for gate in circuit if gate.get('type') in ['CNOT', 'CZ', 'XX', 'ZZ'])

    if num_entangling == 0:
        # Simple circuit - mostly |0...0> state
        bitstring = '0' * num_qubits
        counts[bitstring] = num_shots
    else:
        # Generate a distribution with some spread
        # More entangling gates = more spread in the distribution
        num_outcomes = min(2 ** num_qubits, 2 + num_entangling)
        probs = np.random.dirichlet([1.0] * num_outcomes)

        remaining_shots = num_shots
        for i, prob in enumerate(probs[:-1]):
            bitstring = format(i, f'0{num_qubits}b')
            count = int(prob * num_shots)
            if count > 0:
                counts[bitstring] = count
                remaining_shots -= count

        # Assign remaining shots to last outcome
        if remaining_shots > 0:
            last_bitstring = format(len(probs) - 1, f'0{num_qubits}b')
            counts[last_bitstring] = remaining_shots

    return counts


# -----------------------------------------------------------
# HPC Synergy: Large IonQ Tasks
# -----------------------------------------------------------

class IonQHPCRequest(BaseModel):
    """
    Request model for distributing an IonQ-like benchmark
    (DRB, application, or error mitigation) on HPC.
    """
    job_id: str = Field(..., min_length=1, max_length=100)
    benchmark_type: str = Field(default="drb", pattern="^(drb|application|error_mitigation)$")
    qubits: List[int] = Field(default_factory=list, max_length=30)
    depth: int = Field(default=10, ge=1, le=10000)
    p2q: float = Field(default=0.25, ge=0, le=1)
    # For applications
    app_name: Optional[str] = Field(default=None, max_length=100)
    app_width: Optional[int] = Field(default=None, ge=1)
    # For error mitigation
    circuit: Optional[List[Dict[str, Any]]] = Field(default=None, max_length=10000)
    num_qubits: Optional[int] = Field(default=None, ge=1)

    # HPC resource fields
    cpu_cores: int = Field(default=1, ge=1, le=256)
    gpu_cards: int = Field(default=0, ge=0, le=16)
    memory_gb: float = Field(default=1.0, gt=0, le=1024)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('job_id')
    @classmethod
    def validate_job_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("job_id must contain only alphanumeric characters, underscores, and hyphens")
        return v

    @field_validator('qubits')
    @classmethod
    def validate_qubits(cls, v):
        settings = get_settings()
        for qubit in v:
            if qubit < 0 or qubit >= settings.quantum_max_qubits:
                raise ValueError(f"Qubit index must be between 0 and {settings.quantum_max_qubits - 1}")
        return v

    @field_validator('app_width', 'num_qubits')
    @classmethod
    def validate_qubit_counts(cls, v):
        if v is not None:
            settings = get_settings()
            if v > settings.quantum_max_qubits:
                raise ValueError(f"Value must be <= {settings.quantum_max_qubits}")
        return v


class IonQHPCResponse(BaseModel):
    """
    Response after HPC submission for an IonQ-like task.
    """
    job_id: str
    message: str


@router.post("/submit_hpc_ionq", response_model=IonQHPCResponse)
def submit_hpc_ionq(payload: IonQHPCRequest):
    """
    Submit a large IonQ-like task (DRB, application, or error mitigation)
    to HPC, allocating resources and scheduling an HPC job.

    The HPC job can handle large qubit sets or repeated benchmarks
    that exceed local capabilities.
    """
    coordinator = get_hpc_coordinator()
    resource_manager = get_resource_manager()

    # 1) HPC resource allocation
    request = ResourceRequest(
        job_id=payload.job_id,
        cpu_cores=payload.cpu_cores,
        gpu_cards=payload.gpu_cards,
        memory_gb=payload.memory_gb,
        duration_estimate=3600.0,  # could be from payload.parameters
        metadata=payload.parameters
    )
    allocated = resource_manager.request_resources(request)
    if not allocated:
        raise HTTPException(status_code=400, detail="Insufficient HPC resources for IonQ HPC job.")

    # 2) HPC job creation:
    # We can treat DRB, application, or error mitigation differently in HPC code,
    # or pass them all in parameters to the HPC job's _simulate_cycle logic.
    job_params = {
        "benchmark_type": payload.benchmark_type,
        "qubits": payload.qubits,
        "depth": payload.depth,
        "p2q": payload.p2q,
        "app_name": payload.app_name,
        "app_width": payload.app_width,
        "circuit": payload.circuit,
        "num_qubits": payload.num_qubits,
        **payload.parameters
    }

    # HPCJob usage:
    job = HPCJob(
        job_id=payload.job_id,
        qubit_count=len(payload.qubits) if payload.qubits else (payload.num_qubits or 1),
        code_distance=1,  # IonQ typically doesn't use code distance, but a placeholder
        num_cycles=1,  # You could interpret cycles differently for IonQ tasks
        parameters=job_params
    )

    # 3) Submit HPC job
    try:
        coordinator.submit_job(job)
    except ValueError as e:
        # Release HPC resources if job_id conflict or other error
        resource_manager.release_resources(payload.job_id)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"IonQ HPC job submitted: job_id={payload.job_id}, "
        f"benchmark_type={payload.benchmark_type}, HPC resources allocated."
    )
    return IonQHPCResponse(
        job_id=payload.job_id,
        message=(f"IonQ HPC job '{payload.job_id}' submitted. Track status at /hpc/job_status.")
    )
