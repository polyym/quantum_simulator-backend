# src/routers/quantum_system_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator, Field
from typing import Dict, List, Optional, Union
from uuid import uuid4

import qutip as qt
import numpy as np

from qutip_qip.operations.gates import cnot as cnot_gate

# Existing code from your quantum_system subpackage
from src.quantum_system.network import QuantumNetwork
from src.quantum_system.language import QuantumCompiler, LanguageLevel
from src.quantum_system.error_correction import ErrorCorrection

router = APIRouter()

NETWORKS: Dict[str, QuantumNetwork] = {}
ERROR_CORRECTIONS: Dict[str, ErrorCorrection] = {}

# -------------------------------------------------------------
# Existing endpoints for QuantumNetwork and ErrorCorrection
# -------------------------------------------------------------
@router.post("/network/create")
def create_network(num_nodes: int = 4):
    """
    Create a QuantumNetwork with a specified number of nodes.
    """
    if num_nodes < 1:
        raise HTTPException(status_code=400, detail="num_nodes must be >= 1")

    net = QuantumNetwork(num_nodes=num_nodes)
    net_id = str(uuid4())
    NETWORKS[net_id] = net
    return {"network_id": net_id, "num_nodes": num_nodes}


@router.post("/network/connect")
def connect_nodes(network_id: str, node1: int, node2: int):
    """
    Connect two nodes in an existing QuantumNetwork.
    """
    net = NETWORKS.get(network_id)
    if not net:
        raise HTTPException(status_code=404, detail="Network not found")

    success = net.connect_nodes(node1, node2)
    return {"connected": success}


@router.post("/error_correction/create")
def create_error_correction(num_qubits: int = 1, levels: int = 2):
    """
    Create an ErrorCorrection context (e.g., Steane code).
    """
    ec = ErrorCorrection(num_qubits=num_qubits, levels=levels)
    ec_id = str(uuid4())
    ERROR_CORRECTIONS[ec_id] = ec
    return {"error_correction_id": ec_id, "num_qubits": num_qubits, "levels": levels}

# -------------------------------------------------------------
# New: Simulation Models & Endpoint
# -------------------------------------------------------------
class GateOp(BaseModel):
    """
    Defines a single gate operation.
      e.g. { "gate": "H", "qubits": [0] }
           { "gate": "CNOT", "qubits": [0,1] }
    """
    gate: str
    qubits: List[int]

    @field_validator('qubits')
    def check_qubits(cls, qubits, info):
        """
        Pydantic V2 style validator.
        If gate = "CNOT", ensure we have exactly 2 qubits.
        """
        gate_val = info.data['gate'].upper()
        if gate_val == "CNOT" and len(qubits) != 2:
            raise ValueError("CNOT requires exactly 2 qubits.")
        # You can also enforce single-qubit gates have len(qubits)=1 if you want:
        # elif gate_val in ["H","X","Y","Z"] and len(qubits) != 1:
        #     raise ValueError(f"{gate_val} requires exactly 1 qubit.")
        return qubits


class SimulationRequest(BaseModel):
    """
    Pydantic model for simulation requests.
    Example:
      {
        "num_qubits": 2,
        "gates": [
          { "gate":"H", "qubits":[0] },
          { "gate":"CNOT", "qubits":[0,1] }
        ],
        "shots": 1000
      }
    """
    num_qubits: int = Field(ge=1, le=20)
    gates: List[GateOp]
    shots: Optional[int] = 1000

@router.post("/simulate")
def simulate_quantum_circuit(request: SimulationRequest):
    """
    POST /qsystem/simulate
    Accepts a JSON body matching SimulationRequest.
    1) Initialize all qubits in |0>.
    2) Apply each gate.
    3) Return final state, probabilities, and optional shot-based measurements.
    """
    try:
        # 1) Initialize state
        psi = qt.tensor([qt.basis(2, 0) for _ in range(request.num_qubits)])

        # 2) Apply each gate in sequence
        for gate_op in request.gates:
            psi = apply_gate(psi, gate_op.gate, gate_op.qubits, request.num_qubits)

        # 3) Compute probabilities
        probabilities = measure_probabilities(psi)

        # 4) Perform measurements if requested
        measurements = None
        if request.shots and request.shots > 0:
            measurements = perform_measurements(probabilities, request.shots)

        # 5) Return final state + probabilities
        return {
            "final_state": convert_state_to_json_friendly(psi),
            "probabilities": probabilities,
            "measurements": measurements
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------------------------------------------
# Gate/Measurement Utility Functions
# -------------------------------------------------------------
def apply_gate(state: qt.Qobj, gate_name: str, qubits: List[int], num_qubits: int) -> qt.Qobj:
    """
    Apply a single- or two-qubit gate to the state (QuTiP Qobj).
    gate_name can be: "H", "X", "Y", "Z", "CNOT".
    """
    gate_name_up = gate_name.upper()
    if gate_name_up in ["H","X","Y","Z"]:
        return apply_single_qubit_gate(state, gate_name_up, qubits[0], num_qubits)
    elif gate_name_up == "CNOT":
        return apply_cnot(state, qubits[0], qubits[1], num_qubits)
    else:
        raise ValueError(f"Unsupported gate: {gate_name_up}")

def apply_single_qubit_gate(state: qt.Qobj, gate_name: str, qubit_idx: int, num_qubits: int) -> qt.Qobj:
    """
    Build the operator matrix manually for single-qubit gates (H, X, Y, Z).
    Then expand to the correct qubit index with tensor.
    """
    if gate_name == "H":
        op = (1/np.sqrt(2)) * qt.Qobj([[1, 1], [1, -1]])
    elif gate_name == "X":
        op = qt.sigmax()
    elif gate_name == "Y":
        op = qt.sigmay()
    elif gate_name == "Z":
        op = qt.sigmaz()
    else:
        raise ValueError(f"Unknown single-qubit gate: {gate_name}")

    # Expand to the full system
    expanded_op = qt.tensor([
        op if i == qubit_idx else qt.qeye(2)
        for i in range(num_qubits)
    ])
    return expanded_op * state

def apply_cnot(state: qt.Qobj, control: int, target: int, num_qubits: int) -> qt.Qobj:
    """
    Use cnot from qutip_qip.operations.gates.
    """
    cnot_op = cnot_gate(num_qubits, control, target)
    return cnot_op * state

def measure_probabilities(state: qt.Qobj) -> List[float]:
    """
    Return list of probabilities for each basis state: [p(0...0), p(0...1), ... p(1...1)].
    """
    arr = state.full()
    if state.isket:
        probs = np.abs(arr.flatten())**2
    else:
        # density matrix
        probs = np.real(np.diagonal(arr))
    return probs.tolist()

def perform_measurements(probabilities: List[float], shots: int) -> Dict[str,int]:
    """
    Randomly sample 'shots' times from 'probabilities'.
    Return dict of bitstring -> count.
    """
    results = {}
    dim = len(probabilities)
    from math import log2
    n_qubits = int(log2(dim))

    outcomes = np.random.choice(dim, size=shots, p=probabilities)
    for outcome in outcomes:
        bitstring = format(outcome, f'0{n_qubits}b')
        results[bitstring] = results.get(bitstring, 0) + 1
    return results

def convert_state_to_json_friendly(state: qt.Qobj) -> Union[List[List[float]], List[float]]:
    """
    Convert a QuTiP state (ket or density) into a JSON-friendly array of [real, imag].
    For kets: 1D list of [r,i].
    For density matrices: 2D list of [r,i].
    """
    mat = state.full()
    if state.isket:
        vec = mat.flatten()
        return [[val.real, val.imag] for val in vec]
    else:
        rows, cols = mat.shape
        data = []
        for r in range(rows):
            row_data = []
            for c in range(cols):
                val = mat[r,c]
                row_data.append([val.real, val.imag])
            data.append(row_data)
        return data
