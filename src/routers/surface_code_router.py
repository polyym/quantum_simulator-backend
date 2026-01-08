# src/routers/surface_code_router.py

"""
Router exposing surface code operations (stabilizer measurement, decoding, QEC cycles),
including an optional multi-round HPC-based route for large distances or repeated cycles.
"""

import logging
import re
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional

# Shared services
from src.services import get_hpc_coordinator, get_resource_manager
from src.config import get_settings

# HPC types
from src.quantum_hpc.distributed.coordinator import HPCJob
from src.quantum_hpc.distributed.resource_manager import ResourceRequest

# Surface code classes
from src.quantum_hpc.devices.surface_code.decoder import SurfaceCodeDecoder, DecodedSyndromeResult
from src.quantum_hpc.devices.surface_code.stabilizer import SurfaceCodeStabilizer, StabilizerMeasurementResult

logger = logging.getLogger(__name__)
router = APIRouter()


class MockQuantumProcessor:
    """
    A mock quantum processor for surface code simulations.
    Provides the interface expected by SurfaceCodeStabilizer.
    """

    def __init__(self, num_qubits: int, error_rate: float = 0.01):
        self.num_qubits = num_qubits
        self.error_rate = error_rate

    def apply_gate(self, gate: str, qubits: List[int]) -> None:
        """Apply a gate (mock implementation)."""
        logger.debug(f"MockProcessor: Applying {gate} on qubits {qubits}")

    def measure(self, qubits: List[int], basis: str = "Z") -> tuple:
        """
        Measure qubits and return (results, fidelity).
        Simulates realistic measurement with configurable error rate.
        """
        results = []
        for _ in qubits:
            # Simulate measurement with possible errors
            if np.random.random() < self.error_rate:
                # Error occurred - flip the result
                results.append(1)
            else:
                # No error - return 0 (stable)
                results.append(0)

        fidelity = 1.0 - self.error_rate
        return results, fidelity


# Cached processors and decoders for different distances
_processors: Dict[int, MockQuantumProcessor] = {}
_stabilizers: Dict[int, SurfaceCodeStabilizer] = {}
_decoders: Dict[int, SurfaceCodeDecoder] = {}


def get_processor(distance: int) -> MockQuantumProcessor:
    """Get or create a mock processor for the given distance."""
    if distance not in _processors:
        # Calculate number of qubits needed for surface code
        num_data_qubits = distance * distance
        num_ancilla_qubits = 2 * (distance - 1) * (distance - 1)
        total_qubits = num_data_qubits + num_ancilla_qubits
        _processors[distance] = MockQuantumProcessor(total_qubits)
    return _processors[distance]


def get_stabilizer(distance: int) -> SurfaceCodeStabilizer:
    """Get or create a stabilizer for the given distance."""
    if distance not in _stabilizers:
        processor = get_processor(distance)
        _stabilizers[distance] = SurfaceCodeStabilizer(processor, distance)
    return _stabilizers[distance]


def get_decoder(distance: int) -> SurfaceCodeDecoder:
    """Get or create a decoder for the given distance."""
    if distance not in _decoders:
        _decoders[distance] = SurfaceCodeDecoder(distance)
    return _decoders[distance]


# ---------------------------------------------------
# Pydantic Models with Validation
# ---------------------------------------------------

class StabilizerRequest(BaseModel):
    distance: int = Field(..., ge=1, le=25, description="Surface code distance")
    cycle_index: Optional[int] = Field(default=None, ge=0, le=100000)


class StabilizerResponse(BaseModel):
    X_stabilizers: List[List[int]]
    Z_stabilizers: List[List[int]]
    metadata: Dict[str, Any]


class DecodeRequest(BaseModel):
    distance: int = Field(..., ge=1, le=25, description="Surface code distance")
    stabilizer_data: Dict[str, Any]


class DecodeResponse(BaseModel):
    corrections: Dict[str, Any]
    logical_errors_detected: List[str]


# ---------------------------------------------------
# Single-Round Endpoints
# ---------------------------------------------------

@router.post("/measure_stabilizers", response_model=StabilizerResponse)
async def measure_stabilizers(payload: StabilizerRequest):
    """
    Measure X- and Z-type stabilizers for a given surface code distance,
    returning raw data and metadata.
    """
    try:
        # Get the stabilizer for this distance
        stabilizer = get_stabilizer(payload.distance)

        # Perform stabilizer measurements
        measurement_result: StabilizerMeasurementResult = stabilizer.measure_all_stabilizers(
            cycle_index=payload.cycle_index
        )

        logger.info(f"Measured stabilizers: distance={payload.distance}, cycle={payload.cycle_index}")

        return StabilizerResponse(
            X_stabilizers=measurement_result.X_stabilizers,
            Z_stabilizers=measurement_result.Z_stabilizers,
            metadata=measurement_result.metadata or {"cycle_index": payload.cycle_index, "distance": payload.distance}
        )
    except Exception as e:
        logger.error(f"Stabilizer measurement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stabilizer measurement failed: {str(e)}")


@router.post("/decode_syndrome", response_model=DecodeResponse)
async def decode_syndrome(payload: DecodeRequest):
    """
    Decode a measured syndrome from surface code stabilizers, returning
    corrective operations and any detected logical errors.
    """
    try:
        # Get the decoder for this distance
        decoder = get_decoder(payload.distance)

        # Extract stabilizer data from the payload
        x_stabilizers = payload.stabilizer_data.get("X_stabilizers", [[0] * (payload.distance - 1)] * (payload.distance - 1))
        z_stabilizers = payload.stabilizer_data.get("Z_stabilizers", [[0] * (payload.distance - 1)] * (payload.distance - 1))

        # Create a StabilizerMeasurementResult for the decoder
        syndrome_result = StabilizerMeasurementResult(
            X_stabilizers=x_stabilizers,
            Z_stabilizers=z_stabilizers,
            detection_events=None,
            measurement_fidelity=1.0,
            metadata=payload.stabilizer_data.get("metadata", {"distance": payload.distance})
        )

        # Decode the syndrome
        decoded_result: DecodedSyndromeResult = decoder.decode_syndrome(syndrome_result)

        logger.info(f"Decoded syndrome: distance={payload.distance}, corrections={len(decoded_result.corrections)}")

        return DecodeResponse(
            corrections={"qubit_ops": decoded_result.corrections},
            logical_errors_detected=decoded_result.logical_errors_detected
        )
    except Exception as e:
        logger.error(f"Syndrome decoding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Syndrome decoding failed: {str(e)}")


# ---------------------------------------------------
# Multi-Round QEC Endpoint (HPC Synergy)
# ---------------------------------------------------

class MultiRoundQECRequest(BaseModel):
    """
    Request model for running multiple surface code QEC cycles,
    optionally using HPC resources if you want to scale large distances.
    """
    job_id: str = Field(..., min_length=1, max_length=100)
    distance: int = Field(..., ge=1, le=25, description="Surface code distance")
    rounds: int = Field(default=10, ge=1, le=100000)
    # HPC resource requirements:
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
    coordinator = get_hpc_coordinator()
    resource_manager = get_resource_manager()

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
