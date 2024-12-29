# src/routers/hpc_router.py

from fastapi import APIRouter, HTTPException
from typing import Dict
from uuid import uuid4

# Example imports from HPC subpackage
from src.quantum_hpc.abstract_machine_models import QuantumHPCSystem, AMMType

router = APIRouter()

# In-memory store of HPCSystems
HPC_SYSTEMS: Dict[str, QuantumHPCSystem] = {}


@router.post("/create")
def create_hpc_system(num_nodes: int = 3, model_type: str = "asymmetric"):
    """
    Create a QuantumHPCSystem with a chosen abstract machine model.
    """
    # Validate model_type
    if model_type not in AMMType._value2member_map_:
        raise HTTPException(status_code=400, detail=f"Invalid AMMType: {model_type}")

    system = QuantumHPCSystem(
        num_nodes=num_nodes,
        model_type=AMMType(model_type),
        qubits_per_node=5,         # example default
        classical_bits_per_node=10 # example default
    )

    system_id = str(uuid4())
    HPC_SYSTEMS[system_id] = system

    return {
        "hpc_id": system_id,
        "num_nodes": num_nodes,
        "model_type": model_type,
    }


@router.get("/{hpc_id}/info")
def get_hpc_info(hpc_id: str):
    """
    Retrieve basic info about an existing HPCSystem.
    """
    system = HPC_SYSTEMS.get(hpc_id)
    if not system:
        raise HTTPException(status_code=404, detail="HPC system not found")

    return {
        "hpc_id": hpc_id,
        "num_nodes": system.num_nodes,
        "model_type": system.model_type.value
    }

