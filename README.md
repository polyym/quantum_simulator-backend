# Quantum Simulator Backend

**Version 3.0.0**

A production-ready **FastAPI + QuTiP** backend for quantum computing simulation, featuring:

- **Quantum Circuit Simulation** with full gate support and measurement
- **HPC Job Coordination** for distributed/large-scale simulations
- **IonQ-style Benchmarking** (DRB, application benchmarks, error mitigation)
- **Memristor Gate Acceleration** with power metrics
- **Surface Code Error Correction** with stabilizer measurements and decoding

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Check health
curl http://127.0.0.1:8000/health
```

Open `test_form.html` in your browser for an interactive testing interface.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [API Reference](#api-reference)
4. [Architecture](#architecture)
5. [Configuration](#configuration)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)
8. [Physics Validation](#physics-validation)
9. [Testing](#testing)

---

## Features

### Quantum System Simulation
- **Gates**: X, Y, Z, H, S, T, RX, RY, RZ, PHASE, CNOT, CZ, SWAP, CRZ
- **State Management**: Create, manipulate, and measure quantum systems
- **QuTiP Integration**: Full quantum state vector simulation

### HPC Job Coordination
- Job lifecycle management (QUEUED → RUNNING → COMPLETED/FAILED/CANCELED)
- Resource allocation (CPU cores, GPU cards, memory)
- Background job execution with status tracking

### IonQ Benchmarking
- Direct Randomized Benchmarking (DRB) with proper Clifford sequences
- Single-qubit and two-qubit DRB with recovery gate computation
- Application benchmarks (Hamiltonian simulation, QFT)
- Error mitigation with circuit variants
- Algorithmic Qubit (#AQ) scoring using 1/e threshold

### Memristor Gates ⚠️ *Experimental*
- Parallel gate execution (up to 4 concurrent operations)
- Power metrics (static, dynamic, total energy)
- 4×8 crossbar configuration
- **Note**: This module is speculative/theoretical - see module documentation for details

### Surface Code QEC
- X and Z stabilizer measurements with syndrome history tracking
- MWPM-based syndrome decoding with proper boundary handling
- Detection event computation (XOR of consecutive cycles)
- Multi-round error correction cycles with logical error detection

---

## Installation

### Prerequisites
- Python 3.8+ (3.10 recommended)
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/quantum_simulator-backend.git
cd quantum_simulator-backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.115.6 | Web API framework |
| uvicorn | 0.34.0 | ASGI server |
| qutip | 5.1.0 | Quantum simulation |
| qutip-qip | 0.4.0 | Quantum gates |
| numpy | 2.2.1 | Numerical computing |
| scipy | 1.14.1 | Scientific computing |
| pydantic | 2.10.4 | Data validation |

---

## API Reference

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check API status and version |

### Quantum System (`/qsystem`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/qsystem/create_system` | Create a new quantum system |
| POST | `/qsystem/apply_operation` | Apply a gate operation |
| POST | `/qsystem/measure` | Measure qubits |
| DELETE | `/qsystem/delete_system` | Delete a quantum system |
| POST | `/qsystem/submit_distributed_simulation` | Submit HPC simulation job |

**Example: Create and Measure**
```bash
# Create a 5-qubit system
curl -X POST http://127.0.0.1:8000/qsystem/create_system \
  -H "Content-Type: application/json" \
  -d '{"system_id": "my_system", "num_qubits": 5}'

# Apply Hadamard gate to qubit 0
curl -X POST http://127.0.0.1:8000/qsystem/apply_operation \
  -H "Content-Type: application/json" \
  -d '{"system_id": "my_system", "operation": "H", "qubits": [0]}'

# Apply CNOT gate (control=0, target=1)
curl -X POST http://127.0.0.1:8000/qsystem/apply_operation \
  -H "Content-Type: application/json" \
  -d '{"system_id": "my_system", "operation": "CNOT", "qubits": [0, 1]}'

# Measure qubits 0 and 1
curl -X POST http://127.0.0.1:8000/qsystem/measure \
  -H "Content-Type: application/json" \
  -d '{"system_id": "my_system", "qubits": [0, 1]}'
```

### HPC (`/hpc`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/hpc/submit_job` | Submit an HPC job |
| GET | `/hpc/job_status` | Get job status |
| GET | `/hpc/list_jobs` | List all jobs |
| DELETE | `/hpc/cancel_job` | Cancel a job |
| GET | `/hpc/resources` | Get resource usage |

**Example: Submit HPC Job**
```bash
curl -X POST http://127.0.0.1:8000/hpc/submit_job \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "job123",
    "qubit_count": 10,
    "code_distance": 3,
    "num_cycles": 2,
    "cpu_cores": 4,
    "gpu_cards": 0,
    "memory_gb": 8
  }'

# Check status
curl "http://127.0.0.1:8000/hpc/job_status?job_id=job123"
```

### IonQ Benchmarking (`/ionq`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ionq/drb` | Run Direct Randomized Benchmarking |
| POST | `/ionq/application` | Run application benchmark |
| POST | `/ionq/error_mitigation` | Apply error mitigation |
| POST | `/ionq/submit_hpc_ionq` | Submit HPC IonQ task |

**Example: DRB Benchmark**
```bash
curl -X POST http://127.0.0.1:8000/ionq/drb \
  -H "Content-Type: application/json" \
  -d '{"qubits": [0, 1], "depth": 10, "p2q": 0.25}'
```

### Memristor (`/memristor`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/memristor/run_circuit` | Execute memristor circuit |
| POST | `/memristor/submit_hpc_memristor` | Submit HPC memristor task |

**Example: Run Circuit**
```bash
curl -X POST http://127.0.0.1:8000/memristor/run_circuit \
  -H "Content-Type: application/json" \
  -d '{
    "circuit_id": "mem1",
    "operations": [
      {"gate": "phase", "qubits": [0], "state_dim": 2},
      {"gate": "swap", "qubits": [0, 1]}
    ]
  }'
```

### Surface Code (`/surface_code`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/surface_code/measure_stabilizers` | Measure stabilizers |
| POST | `/surface_code/decode_syndrome` | Decode syndrome |
| POST | `/surface_code/run_multi_round_qec` | Run multi-round QEC |

**Example: Measure Stabilizers**
```bash
curl -X POST http://127.0.0.1:8000/surface_code/measure_stabilizers \
  -H "Content-Type: application/json" \
  -d '{"distance": 3, "cycle_index": 1}'
```

---

## Architecture

```
quantum_simulator-backend/
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies
├── test_form.html                   # Interactive testing interface
├── start.sh                         # Deployment script
└── src/
    ├── routers/                     # API endpoint handlers
    │   ├── quantum_system_router.py
    │   ├── hpc_router.py
    │   ├── ionq_router.py
    │   ├── memristor_router.py
    │   └── surface_code_router.py
    ├── quantum_system/              # Core quantum simulation
    │   ├── error_correction.py
    │   ├── language.py
    │   └── network.py
    ├── quantum_hpc/                 # HPC infrastructure
    │   ├── abstract/                # Base classes
    │   ├── devices/                 # QEC codes (surface code)
    │   ├── distributed/             # Job coordination
    │   ├── hardware/                # Noise, calibration, topology
    │   └── virtualization/          # Simulation engines
    ├── ionq_benchmarking/           # IonQ-style benchmarks
    ├── memristor_gates/             # Memristor acceleration
    └── utils/                       # Shared utilities
        ├── benchmarking.py
        ├── error_analysis.py
        ├── metrics_collection.py
        └── visualization.py
```

### Request Flow

1. **HTTP Request** → FastAPI router validates input
2. **Router** → Delegates to domain-specific module
3. **Module** → Executes quantum operations via QuTiP
4. **Response** → JSON result returned to client

### HPC Job Flow

1. **Submit** → Job queued, resources allocated
2. **Execute** → Background thread runs simulation
3. **Complete** → Results stored, resources released
4. **Query** → Client polls for status/results

---

## Configuration

### Resource Manager

Default HPC resources (in `quantum_system_router.py`):
```python
resource_manager = ResourceManager(
    total_cores=64,
    total_gpus=4,
    total_memory_gb=128.0
)
```

### CORS Settings

Default allows all origins (development). For production, restrict in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Deployment

### Local Development

```bash
uvicorn main:app --reload --port 8000
```

### Production (Docker)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud (Render, Railway, etc.)

Use the provided `start.sh`:
```bash
#!/usr/bin/env bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: qutip` | Run `pip install -r requirements.txt` |
| `Port already in use` | Kill existing process or use different port |
| `CORS errors` | Check `allow_origins` in `main.py` |
| `Insufficient HPC resources` | Reduce CPU/GPU/memory request |
| `System not found` | Create system before applying operations |

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Tips

- Keep qubit count under 20 for reasonable simulation times
- Use HPC endpoints for large circuits
- Batch multiple operations before measuring

---

## API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## Physics Implementation Notes

This section documents the physics accuracy of the quantum simulation components.

### Quantum Gates ✓
- All single-qubit gates (X, Y, Z, H, S, T, RX, RY, RZ, PHASE) use correct unitary matrices
- Two-qubit gates (CNOT, CZ, SWAP, CRZ) properly handle non-adjacent qubits
- Gate expansion to full Hilbert space preserves entanglement structure

### Measurement ✓
- Born's rule correctly implemented: P(outcome) = |⟨ψ|basis⟩|²
- Partial measurement with proper state renormalization
- Support for X, Y, Z measurement bases

### Noise Models ✓
- **Depolarizing**: Correct Kraus operators ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
- **Amplitude Damping**: Proper T1 relaxation model
- **Phase Damping**: Dephasing with correct T2 effects
- **Thermal Noise**: Uses physical units (Kelvin, GHz, μs) with Bose-Einstein statistics
- **Crosstalk (ZZ)**: Full ZZ interaction Hamiltonian H = Σ ζ_{ij} Z_i ⊗ Z_j with proper phase accumulation

### Surface Code QEC ✓
- Syndrome history tracking for proper detection event computation
- MWPM-based decoder with boundary node handling
- Physical error chain identification
- Logical error detection across code boundaries

### IonQ Benchmarking ✓
- DRB uses random Clifford gate sequences (24 single-qubit Cliffords)
- Recovery gate (inverse Clifford) properly computed
- Error per Clifford extracted from exponential decay fit
- Hellinger fidelity calculation matches IonQ methodology
- **Application Benchmarks**: Use error-model-based fidelity estimation: F ≈ (1-ε₁q)^n₁q × (1-ε₂q)^n₂q

### Memristor Module ⚠️
- **EXPERIMENTAL**: No validated physics for quantum-memristor coupling
- Power estimates are theoretical projections
- Use for architectural exploration only
- API responses include explicit experimental warnings

### Scalability Limits
- **Practical limit**: 20 qubits (state vector simulation scales as O(2^n))
- **Maximum supported**: 25 qubits (configurable)
- **Memory scaling**: ~16 bytes × 2^n (e.g., 20 qubits = 16 MB, 25 qubits = 512 MB)
- For larger simulations, use HPC endpoints or consider tensor network methods

---

## Physics Validation

This codebase includes a comprehensive physics validation test suite to ensure scientific accuracy.

### Validation Categories

| Category | Tests | Description |
|----------|-------|-------------|
| Gate Unitarity | 14 | Verifies U†U = I for all gate matrices |
| State Normalization | 4 | Ensures ⟨ψ\|ψ⟩ = 1 after operations |
| Born's Rule | 3 | Validates P(outcome) = \|⟨ψ\|basis⟩\|² |
| Noise Channels | 5 | Checks Kraus operator trace preservation |
| Known Circuits | 4 | Tests Bell states, GHZ, QFT properties |
| Clifford Properties | 3 | Validates Clifford group closure |
| Scalability | 2 | Memory estimation and qubit limits |

### Running Physics Tests

```bash
# Run all physics validation tests
python -m pytest tests/test_physics_validation.py -v

# Run specific category
python -m pytest tests/test_physics_validation.py -v -k "TestGateUnitarity"

# Run with coverage
python -m pytest tests/test_physics_validation.py --cov=src
```

### Test Results Summary

All 35 physics validation tests pass, confirming:
- All quantum gates are unitary (U†U = I)
- State vectors remain normalized after operations
- Measurement probabilities follow Born's rule
- Noise channels preserve trace (Σ E†E = I)
- Known quantum circuits produce correct results
- Clifford gates form a valid group

---

## Testing

### Test Suite Structure

```
tests/
├── __init__.py
└── test_physics_validation.py    # 35 physics validation tests
```

### Test Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run with detailed output
python -m pytest tests/ -v --tb=short

# Run specific test class
python -m pytest tests/test_physics_validation.py::TestGateUnitarity -v
```

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

For questions or issues, open a GitHub issue.
