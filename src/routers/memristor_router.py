# src/routers/memristor_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
from uuid import uuid4

from src.memristor_gates.enhanced_gates import (
    ParallelQuantumMemristorAccelerator
)

router = APIRouter()

ACCELERATORS: Dict[str, ParallelQuantumMemristorAccelerator] = {}

class MemristorRequest(BaseModel):
    accelerator_id: str
    circuit: List[Dict[str, Any]]

@router.post("/accelerator/create")
def create_memristor_accelerator(max_parallel_ops: int = 4):
    """
    Create a ParallelQuantumMemristorAccelerator for executing circuits with memristor-based gates.
    """
    accelerator = ParallelQuantumMemristorAccelerator(max_parallel_ops=max_parallel_ops)
    accel_id = str(uuid4())
    ACCELERATORS[accel_id] = accelerator
    return {"accelerator_id": accel_id, "max_parallel_ops": max_parallel_ops}

@router.post("/accelerator/execute_circuit")
async def execute_circuit(req: MemristorRequest):
    """
    Expects JSON like:
      {
        "accelerator_id": "...",
        "circuit": [
          { "gate": "HADAMARD", "qubits": [0] },
          { "gate": "SWAP", "qubits": [0,1] }
        ]
      }
    """
    accel = ACCELERATORS.get(req.accelerator_id)
    if not accel:
        raise HTTPException(status_code=404, detail="Accelerator not found")

    state_vector, metrics = await accel.execute_quantum_circuit(req.circuit)
    return {
        "final_state": state_vector.tolist() if state_vector is not None else None,
        "metrics": metrics
    }
