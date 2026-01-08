# src/routers/quantum_system_router.py

"""
Router exposing quantum system operations, including circuit creation,
applying gates, and measurement. Optionally integrates with HPC for
large or distributed quantum simulations.
"""

import logging
import re
import numpy as np
import qutip as qt
from qutip_qip.operations import gates as qip_gates
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

# Shared services
from src.services import get_hpc_coordinator, get_resource_manager, get_quantum_system_manager
from src.config import get_settings

# HPC types
from src.quantum_hpc.distributed.coordinator import HPCJob
from src.quantum_hpc.distributed.resource_manager import ResourceRequest

logger = logging.getLogger(__name__)
router = APIRouter()


class QuantumSimulationEngine:
    """
    A QuTiP-based quantum simulation engine for local circuit execution.
    """

    def __init__(self, num_qubits: int, description: Optional[str] = None):
        self.num_qubits = num_qubits
        self.description = description
        # Initialize state as |00...0>
        self.state = qt.tensor([qt.basis(2, 0) for _ in range(num_qubits)])

        # Define standard gates
        self.gates = {
            'I': qt.qeye(2),
            'X': qt.sigmax(),
            'Y': qt.sigmay(),
            'Z': qt.sigmaz(),
            'H': qt.Qobj([[1, 1], [1, -1]]) / np.sqrt(2),
            'S': qt.Qobj([[1, 0], [0, 1j]]),
            'T': qt.Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
            'CNOT': qip_gates.cnot(),
            'CZ': qip_gates.cphase(np.pi),
            'SWAP': qip_gates.swap(),
        }
        logger.debug(f"QuantumSimulationEngine initialized with {num_qubits} qubits")

    def _apply_single_qubit_gate(self, gate: qt.Qobj, qubit: int) -> None:
        """Apply a single-qubit gate to a specific qubit."""
        ops = [qt.qeye(2) for _ in range(self.num_qubits)]
        ops[qubit] = gate
        full_gate = qt.tensor(ops)
        self.state = full_gate * self.state

    def _apply_two_qubit_gate(self, gate: qt.Qobj, control: int, target: int) -> None:
        """Apply a two-qubit gate (control, target)."""
        # Build permutation to bring control and target to positions 0 and 1
        if self.num_qubits == 2:
            if control == 0 and target == 1:
                full_gate = gate
            else:
                # Swap and apply
                full_gate = qt.swap() * gate * qt.swap()
        else:
            # For more qubits, we need to construct the full operator
            # Using QuTiP's gate_expand functions
            if gate.dims == [[4], [4]] or gate.dims == [[2, 2], [2, 2]]:
                # Two-qubit gate
                full_gate = self._expand_two_qubit_gate(gate, control, target)
            else:
                raise ValueError(f"Unsupported gate dimensions: {gate.dims}")

        self.state = full_gate * self.state

    def _expand_two_qubit_gate(self, gate: qt.Qobj, control: int, target: int) -> qt.Qobj:
        """
        Expand a two-qubit gate to the full Hilbert space.

        For non-adjacent qubits, this uses SWAP-based routing to bring qubits
        together, apply the gate, then swap back. This preserves the entangling
        nature of two-qubit gates which cannot be decomposed via partial trace.

        Physics Note: Two-qubit entangling gates (like CNOT) create quantum
        correlations that cannot be represented as tensor products of single-qubit
        operations. The previous ptrace-based approach was incorrect as it
        destroyed entanglement structure.
        """
        n = self.num_qubits

        # Reshape gate if needed (convert from [[4],[4]] to [[2,2],[2,2]] dims)
        if gate.dims == [[4], [4]]:
            gate = qt.Qobj(gate.full().reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4),
                          dims=[[2, 2], [2, 2]])

        # For adjacent qubits, use direct tensor product expansion
        if abs(control - target) == 1:
            q1, q2 = min(control, target), max(control, target)

            # If control > target, we need to swap the gate's qubit ordering
            if control > target:
                swap_op = qt.Qobj([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                                 dims=[[2, 2], [2, 2]])
                gate_ordered = swap_op * gate * swap_op
            else:
                gate_ordered = gate

            # Build full operator via tensor product
            parts = []
            if q1 > 0:
                parts.append(qt.tensor([qt.qeye(2) for _ in range(q1)]))
            parts.append(gate_ordered)
            if q2 < n - 1:
                parts.append(qt.tensor([qt.qeye(2) for _ in range(n - q2 - 1)]))

            if len(parts) == 1:
                return parts[0]
            else:
                return qt.tensor(parts)

        # For non-adjacent qubits, use explicit matrix construction
        # This correctly handles the entanglement structure by directly computing
        # the action of the gate on the appropriate subspace
        dim = 2 ** n
        full_matrix = np.zeros((dim, dim), dtype=complex)
        gate_matrix = gate.full()

        for i in range(dim):
            for j in range(dim):
                # Extract bits for all qubits (big-endian: qubit 0 is MSB)
                bits_i = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                bits_j = [(j >> (n - 1 - k)) & 1 for k in range(n)]

                # Check if all qubits OTHER than control and target match
                other_match = all(bits_i[k] == bits_j[k] for k in range(n)
                                 if k != control and k != target)

                if other_match:
                    # Map to 2-qubit gate indices: |control, target⟩ → index
                    # Index in 2-qubit space: control_bit * 2 + target_bit
                    i2 = bits_i[control] * 2 + bits_i[target]
                    j2 = bits_j[control] * 2 + bits_j[target]
                    full_matrix[i, j] = gate_matrix[i2, j2]
                # else: matrix element is 0 (already initialized)

        return qt.Qobj(full_matrix, dims=[[2]*n, [2]*n])

    def apply_gate(self, operation: str, qubits: List[int], params: Dict[str, Any] = None) -> None:
        """Apply a quantum gate operation."""
        params = params or {}

        if len(qubits) == 1:
            qubit = qubits[0]
            if qubit < 0 or qubit >= self.num_qubits:
                raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")

            if operation.upper() in self.gates:
                gate = self.gates[operation.upper()]
                self._apply_single_qubit_gate(gate, qubit)
            elif operation.upper() == 'RX':
                angle = params.get('angle', np.pi)
                gate = qip_gates.rx(angle)
                self._apply_single_qubit_gate(gate, qubit)
            elif operation.upper() == 'RY':
                angle = params.get('angle', np.pi)
                gate = qip_gates.ry(angle)
                self._apply_single_qubit_gate(gate, qubit)
            elif operation.upper() == 'RZ':
                angle = params.get('angle', np.pi)
                gate = qip_gates.rz(angle)
                self._apply_single_qubit_gate(gate, qubit)
            elif operation.upper() == 'PHASE' or operation.upper() == 'P':
                angle = params.get('angle', np.pi/2)
                gate = qt.Qobj([[1, 0], [0, np.exp(1j * angle)]])
                self._apply_single_qubit_gate(gate, qubit)
            else:
                raise ValueError(f"Unknown single-qubit operation: {operation}")

        elif len(qubits) == 2:
            control, target = qubits[0], qubits[1]
            if control < 0 or control >= self.num_qubits or target < 0 or target >= self.num_qubits:
                raise ValueError(f"Qubit indices out of range")
            if control == target:
                raise ValueError("Control and target qubits must be different")

            if operation.upper() == 'CNOT' or operation.upper() == 'CX':
                gate = self.gates['CNOT']
                self._apply_two_qubit_gate(gate, control, target)
            elif operation.upper() == 'CZ':
                gate = self.gates['CZ']
                self._apply_two_qubit_gate(gate, control, target)
            elif operation.upper() == 'SWAP':
                gate = self.gates['SWAP']
                self._apply_two_qubit_gate(gate, control, target)
            elif operation.upper() == 'CRZ':
                angle = params.get('angle', np.pi)
                # Controlled-RZ gate
                gate = qt.Qobj([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, np.exp(1j * angle)]],
                              dims=[[2, 2], [2, 2]])
                self._apply_two_qubit_gate(gate, control, target)
            else:
                raise ValueError(f"Unknown two-qubit operation: {operation}")
        else:
            raise ValueError(f"Operations on {len(qubits)} qubits not supported")

        logger.debug(f"Applied {operation} on qubits {qubits}")

    def measure_probabilities(self, qubits: List[int] = None) -> Dict[str, float]:
        """
        Get measurement probabilities for specified qubits.
        If qubits is None, measure all qubits.
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))

        # Get the state vector
        if self.state.isket:
            state_vector = self.state.full().flatten()
        else:
            # For density matrix, get diagonal
            state_vector = np.sqrt(np.abs(np.diag(self.state.full())))

        probabilities = np.abs(state_vector) ** 2

        # Build outcome dictionary
        outcomes = {}
        for i, prob in enumerate(probabilities):
            if prob > 1e-10:
                bitstring = format(i, f'0{self.num_qubits}b')
                # Extract only the measured qubits
                measured_bits = ''.join(bitstring[q] for q in qubits)
                if measured_bits in outcomes:
                    outcomes[measured_bits] += prob
                else:
                    outcomes[measured_bits] = prob

        return outcomes

    def get_state_vector(self) -> List[complex]:
        """Return the current state vector as a list of complex amplitudes."""
        if self.state.isket:
            return self.state.full().flatten().tolist()
        else:
            return np.diag(self.state.full()).tolist()


# -------------------------------------------------------------------------
# Pydantic models with validation
# -------------------------------------------------------------------------
class CreateSystemRequest(BaseModel):
    system_id: str = Field(..., min_length=1, max_length=100, description="Unique system identifier")
    num_qubits: int = Field(..., ge=1, description="Number of qubits in the system")
    description: Optional[str] = Field(None, max_length=500, description="Optional system description")

    @field_validator('num_qubits')
    @classmethod
    def validate_num_qubits(cls, v):
        settings = get_settings()
        if v > settings.quantum_max_qubits:
            raise ValueError(f"num_qubits must be <= {settings.quantum_max_qubits}")
        return v

    @field_validator('system_id')
    @classmethod
    def validate_system_id(cls, v):
        # Only allow alphanumeric, underscore, and hyphen
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("system_id must contain only alphanumeric characters, underscores, and hyphens")
        return v


class CreateSystemResponse(BaseModel):
    system_id: str
    message: str
    scalability_info: Dict[str, Any] = {
        "max_practical_qubits": 20,
        "max_supported_qubits": 25,
        "memory_per_qubit_doubling": "2x",
        "note": "State vector simulation scales as O(2^n). For n>20 qubits, "
                "consider using tensor network methods or HPC resources."
    }


class ApplyOperationRequest(BaseModel):
    system_id: str = Field(..., min_length=1, max_length=100)
    operation: str = Field(..., min_length=1, max_length=20, description="Gate operation name")
    qubits: List[int] = Field(..., min_length=1, max_length=3, description="Target qubit indices")
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('qubits')
    @classmethod
    def validate_qubits(cls, v):
        settings = get_settings()
        for qubit in v:
            if qubit < 0 or qubit >= settings.quantum_max_qubits:
                raise ValueError(f"Qubit index must be between 0 and {settings.quantum_max_qubits - 1}")
        return v


class ApplyOperationResponse(BaseModel):
    message: str


class MeasureRequest(BaseModel):
    system_id: str = Field(..., min_length=1, max_length=100)
    qubits: List[int] = Field(..., min_length=1, description="Qubits to measure")

    @field_validator('qubits')
    @classmethod
    def validate_qubits(cls, v):
        settings = get_settings()
        for qubit in v:
            if qubit < 0 or qubit >= settings.quantum_max_qubits:
                raise ValueError(f"Qubit index must be between 0 and {settings.quantum_max_qubits - 1}")
        return v


class MeasureResponse(BaseModel):
    outcomes: Dict[str, float]


# -------------------------------------------------------------------------
# Local Endpoints (Small-Scale or Non-HPC)
# -------------------------------------------------------------------------

def _estimate_memory_mb(num_qubits: int) -> float:
    """Estimate memory required for state vector in megabytes."""
    # State vector: 2^n complex128 values, each 16 bytes
    return (2 ** num_qubits * 16) / (1024 * 1024)


@router.post("/create_system", response_model=CreateSystemResponse)
def create_system(payload: CreateSystemRequest):
    """
    Create a new local quantum system instance (e.g., a small simulation).
    For HPC synergy, see /submit_distributed_simulation.

    Note: State vector simulation scales exponentially with qubit count.
    Memory requirements: O(2^n) where n = number of qubits.
    - 10 qubits: ~16 KB
    - 15 qubits: ~0.5 MB
    - 20 qubits: ~16 MB
    - 25 qubits: ~512 MB
    - 30 qubits: ~16 GB

    For systems with >20 qubits, consider using /submit_distributed_simulation
    with HPC resources.
    """
    qsm = get_quantum_system_manager()
    settings = get_settings()

    if qsm.exists(payload.system_id):
        raise HTTPException(status_code=400, detail="System ID already exists.")

    # Add warning for large qubit counts
    memory_mb = _estimate_memory_mb(payload.num_qubits)
    warning_msg = ""
    if payload.num_qubits > settings.quantum_practical_qubits:
        warning_msg = (
            f" WARNING: {payload.num_qubits} qubits requires ~{memory_mb:.1f} MB. "
            f"Consider using HPC for systems > {settings.quantum_practical_qubits} qubits."
        )
        logger.warning(
            f"Creating large quantum system: {payload.num_qubits} qubits, "
            f"~{memory_mb:.1f} MB memory required"
        )

    try:
        engine = QuantumSimulationEngine(
            num_qubits=payload.num_qubits,
            description=payload.description
        )

        if not qsm.create_system(payload.system_id, engine, payload.description):
            raise HTTPException(
                status_code=503,
                detail="Maximum number of quantum systems reached. Please delete unused systems."
            )

        logger.info(f"Created quantum system '{payload.system_id}' with {payload.num_qubits} qubits")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create quantum system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create quantum system: {str(e)}")

    return CreateSystemResponse(
        system_id=payload.system_id,
        message=f"Quantum system '{payload.system_id}' created with {payload.num_qubits} qubits.{warning_msg}",
        scalability_info={
            "num_qubits": payload.num_qubits,
            "estimated_memory_mb": round(memory_mb, 2),
            "max_practical_qubits": settings.quantum_practical_qubits,
            "max_supported_qubits": settings.quantum_max_qubits,
            "note": "State vector simulation scales as O(2^n). For n>20 qubits, "
                    "consider using tensor network methods or HPC resources."
        }
    )


@router.post("/apply_operation", response_model=ApplyOperationResponse)
def apply_operation(payload: ApplyOperationRequest):
    """
    Apply a quantum operation (gate or error correction step) to a local system.
    """
    qsm = get_quantum_system_manager()
    engine = qsm.get_system(payload.system_id)

    if engine is None:
        raise HTTPException(status_code=404, detail="System not found.")

    try:
        engine.apply_gate(payload.operation, payload.qubits, payload.params)
        logger.debug(f"Applied {payload.operation} on system '{payload.system_id}', qubits {payload.qubits}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to apply operation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply operation: {str(e)}")

    return ApplyOperationResponse(
        message=(
            f"Operation '{payload.operation}' applied on system '{payload.system_id}' "
            f"for qubits {payload.qubits}."
        )
    )


@router.post("/measure", response_model=MeasureResponse)
def measure(payload: MeasureRequest):
    """
    Measure specified qubits in the local system and return outcome probabilities.
    """
    qsm = get_quantum_system_manager()
    engine = qsm.get_system(payload.system_id)

    if engine is None:
        raise HTTPException(status_code=404, detail="System not found.")

    try:
        results = engine.measure_probabilities(payload.qubits)
        logger.debug(f"Measured system '{payload.system_id}', qubits {payload.qubits}: {results}")
    except Exception as e:
        logger.error(f"Failed to measure: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to measure: {str(e)}")

    return MeasureResponse(outcomes=results)


@router.delete("/delete_system")
def delete_system(system_id: str):
    """
    Delete a local quantum system from memory.
    """
    qsm = get_quantum_system_manager()

    if qsm.delete_system(system_id):
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
    job_id: str = Field(..., min_length=1, max_length=100)
    num_qubits: int = Field(..., ge=1)
    code_distance: int = Field(default=3, ge=1, le=25)
    num_cycles: int = Field(default=1, ge=1, le=10000)
    cpu_cores: int = Field(default=1, ge=1, le=256)
    gpu_cards: int = Field(default=0, ge=0, le=16)
    memory_gb: float = Field(default=1.0, gt=0, le=1024)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('num_qubits')
    @classmethod
    def validate_num_qubits(cls, v):
        settings = get_settings()
        if v > settings.quantum_max_qubits:
            raise ValueError(f"num_qubits must be <= {settings.quantum_max_qubits}")
        return v

    @field_validator('job_id')
    @classmethod
    def validate_job_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("job_id must contain only alphanumeric characters, underscores, and hyphens")
        return v


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
    coordinator = get_hpc_coordinator()
    resource_manager = get_resource_manager()

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
