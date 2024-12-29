# src/routers/ionq_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from uuid import uuid4

# Imports from IonQ subpackage
from src.ionq_benchmarking.core import IonQDevice, ApplicationBenchmarks
from src.ionq_benchmarking.error_mitigation import ErrorMitigation, CircuitOptimizer
from src.ionq_benchmarking.timing import TimingAnalyzer

router = APIRouter()

IONQ_DEVICES: Dict[str, IonQDevice] = {}
BENCHMARKS: Dict[str, ApplicationBenchmarks] = {}

# -------------------------------------------------------------
# 1) Create IonQ Device Endpoint
# -------------------------------------------------------------
@router.post("/device/create")
def create_ionq_device(num_qubits: int = 30):
    """
    Create an IonQDevice with a given number of qubits.
    Returns a JSON response with the new device_id.
    """
    device = IonQDevice(num_qubits=num_qubits)
    dev_id = str(uuid4())
    IONQ_DEVICES[dev_id] = device
    return {"device_id": dev_id, "num_qubits": num_qubits}

# -------------------------------------------------------------
# 2) Run IonQ Application Benchmark
# -------------------------------------------------------------
@router.post("/benchmark/run")
def run_ionq_benchmark(device_id: str, name: str, width: int = 2):
    """
    Run an application benchmark (e.g., 'hamiltonian_simulation', 'phase_estimation', etc.)
    on an existing IonQ device.
    
    This expects query parameters or form data by default:
      device_id=<...>&name=<...>&width=<...>
    If you want JSON body instead, make it a Pydantic model similarly to DRBRequest.
    """
    device = IONQ_DEVICES.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="IonQ device not found")

    bench_id = f"bench_{device_id}"
    if bench_id not in BENCHMARKS:
        BENCHMARKS[bench_id] = ApplicationBenchmarks(device=device)

    app_bench = BENCHMARKS[bench_id]
    fidelity = app_bench.run_benchmark(name, width)

    return {"benchmark": name, "width": width, "fidelity": fidelity}

# -------------------------------------------------------------
# 3) DRB Request Model
# -------------------------------------------------------------
class DRBRequest(BaseModel):
    """
    Pydantic model for IonQ DRB requests.
    Matches the JSON structure your test_form.html sends:
      {
        "device_id": "<some-string>",
        "qubits": [0,1],
        "depth": 10,
        "p2q": 0.25
      }
    """
    device_id: str
    qubits: List[int]
    depth: int = 10
    p2q: float = 0.25

# -------------------------------------------------------------
# 4) Run Direct Randomized Benchmarking
# -------------------------------------------------------------
@router.post("/device/drb")
def run_direct_randomized_benchmarking(req: DRBRequest):
    """
    Simple demonstration of IonQDevice.run_drb, now using a JSON body.
    Expects:
      {
        "device_id": "...",
        "qubits": [...],
        "depth": 10,
        "p2q": 0.25
      }
    """
    device = IONQ_DEVICES.get(req.device_id)
    if not device:
        raise HTTPException(status_code=404, detail="IonQ device not found")

    success_prob = device.run_drb(
        qubits=req.qubits,
        depth=req.depth,
        p2q=req.p2q
    )
    return {"device_id": req.device_id, "success_probability": success_prob}
