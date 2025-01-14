# Quantum Simulator Backend

This project is a **FastAPI + QuTiP** backend designed to handle a variety of quantum computing functionalities:

- **Quantum System Simulation** (general gates, noise models, measurement)  
- **HPC** (High-Performance Computing) quantum job submission & resource management  
- **IonQ**-style benchmarking (DRB, application benchmarks, etc.)  
- **Memristor-based** quantum gate accelerators (parallel gate modeling)  
- **Surface Code** error correction (stabilizers, decoding, multi-round QEC)

All features are exposed via **HTTP endpoints** for easy integration with frontends or external clients. The code uses a **router-based design**, with each domain (HPC, IonQ, Memristor, surface code, and quantum system) in separate files.

---

## Table of Contents

1. [Features](#features)  
2. [Architecture](#architecture)  
3. [Repository Structure](#repository-structure)  
4. [Requirements](#requirements)  
5. [Installation](#installation)  
6. [Running the Server](#running-the-server)  
7. [Usage](#usage)  
8. [Endpoints](#endpoints)  
9. [Troubleshooting](#troubleshooting)  
10. [License](#license)

---

## Features

- **Quantum System**  
  - Single- and multi-qubit gates (X, Y, Z, Phase, T, Swap, CNOT, etc.)  
  - Optional noise modeling (amplitude/phase damping, depolarizing)  
  - Local or HPC-based simulation, returning final states or measurement outcomes  

- **HPC**  
  - Job coordination (`coordinator.py`) with states (*QUEUED, RUNNING, COMPLETED, FAILED, CANCELED*)  
  - Resource management (`resource_manager.py`) for CPU/GPU/memory requests  
  - HPC synergy in each router to handle large/distributed tasks  

- **IonQ**  
  - IonQ device simulation (via QuTiP or placeholders)  
  - DRB (Direct Randomized Benchmarking), application benchmarks (QFT, Hamiltonian sim), error mitigation  
  - HPC endpoints for big IonQ tasks or repeated DRB  

- **Memristor Gates**  
  - Parallel quantum ops on memristor crossbars  
  - Power metrics (static/dynamic power, total energy)  
  - HPC synergy for large or multi-node memristor tasks  

- **Surface Code**  
  - Single-round stabilizer measure and decode  
  - Multi-round QEC HPC endpoint for distance scaling, repeated cycles  

- **Router-Based Design**  
  - Distinct URL prefixes for HPC (`/hpc`), IonQ (`/ionq`), Memristor (`/memristor`), Quantum System (`/qsystem`), Surface Code (`/surface_code`)  
  - Maintains code clarity and easy expansions

---

## Architecture

A **high-level overview** of how modules fit together:

1. **FastAPI Application**  
   - **`main.py`**: The entry point that includes each router (HPC, IonQ, Memristor, quantum system, surface code).  
   - Global exception handlers for uniform error responses.

2. **Quantum System** (`src/quantum_system/`)  
   - Local or HPC-based simulation logic (e.g., `error_correction.py`, `language.py`, `network.py`).  
   - **`quantum_system_router.py`**: endpoints for create/apply/measure and HPC distributed simulation.

3. **Quantum HPC** (`src/quantum_hpc/`)  
   - **`distributed/`**: HPC job coordinator, resource manager, synchronization.  
   - **`devices/`**: advanced QEC codes (e.g., surface code).  
   - **`hardware/`**: noise models, calibration, topologies.  
   - **`virtualization/`**: bridging HPC tasks with quantum logic.  
   - **`hpc_router.py`**: HPC job submission, job status checks, resource usage.

4. **IonQ Benchmarking** (`src/ionq_benchmarking/`)  
   - IonQ-like DRB, application benchmarks, error mitigation.  
   - HPC synergy for large circuits or repeated tasks in `ionq_router.py`.

5. **Memristor Gates** (`src/memristor_gates/`)  
   - `enhanced_gates.py`: parallel memristor crossbar classes.  
   - `memristor_router.py`: local runs vs HPC synergy for big circuits.

6. **Surface Code**  
   - Found in `quantum_hpc/devices/surface_code/` (stabilizer, decoder, logical ops).  
   - `surface_code_router.py`: local or HPC multi-round QEC.

7. **Utilities** (`src/utils/`)  
   - Shared code: error analysis, metrics collection, benchmarking, visualization.

### Advanced Architecture Insights

#### Module Interaction Flow

The quantum simulator backend uses a carefully designed interaction model:

1. **Request Routing**
   - Incoming HTTP requests are first processed by FastAPI routers
   - Routers validate input and delegate to appropriate domain-specific modules

2. **Quantum System Simulation Flow**
   ```python
   def simulate_quantum_system(system_config):
       # Validate system configuration
       validated_config = validate_config(system_config)
       
       # Initialize quantum system
       quantum_system = QuantumSystem(validated_config)
       
       # Apply quantum operations
       for operation in validated_config.operations:
           quantum_system.apply_operation(operation)
       
       # Perform measurement
       measurement_result = quantum_system.measure()
       
       return measurement_result
   ```

3. **HPC Resource Allocation**
   - Dynamic resource allocation based on:
     - Available computational resources
     - Complexity of quantum simulation
     - User-defined constraints

---

## Repository Structure

```
quantum_simulator-backend/
├── main.py
├── requirements.txt
├── README.md
├── test_form.html
├── start.sh
└── src/
    ├── routers/
    │   ├── __init__.py
    │   ├── quantum_system_router.py
    │   ├── hpc_router.py
    │   ├── ionq_router.py
    │   ├── memristor_router.py
    │   └── surface_code_router.py
    ├── quantum_system/
    │   ├── __init__.py
    │   ├── error_correction.py
    │   ├── language.py
    │   └── network.py
    ├── quantum_hpc/
    │   ├── __init__.py
    │   ├── abstract/
    │   │   ├── __init__.py
    │   │   ├── quantum_processor.py
    │   │   ├── error_correction.py
    │   │   └── interconnect.py
    │   ├── devices/
    │   │   ├── __init__.py
    │   │   ├── surface_code/
    │   │   │   ├── __init__.py
    │   │   │   ├── stabilizer.py
    │   │   │   ├── decoder.py
    │   │   │   └── logical_ops.py
    │   │   ├── bacon_shor/...
    │   │   └── color_code/...
    │   ├── distributed/
    │   │   ├── __init__.py
    │   │   ├── coordinator.py
    │   │   ├── resource_manager.py
    │   │   └── synchronization.py
    │   ├── hardware/
    │   │   ├── __init__.py
    │   │   ├── topology.py
    │   │   ├── noise_model.py
    │   │   └── calibration.py
    │   └── virtualization/
    │       ├── __init__.py
    │       ├── simulation.py
    │       └── emulation.py
    ├── ionq_benchmarking/
    │   ├── __init__.py
    │   ├── core.py
    │   ├── error_mitigation.py
    │   └── timing.py
    ├── memristor_gates/
    │   ├── __init__.py
    │   └── enhanced_gates.py
    └── utils/
        ├── __init__.py
        ├── benchmarking.py
        ├── error_analysis.py
        ├── metrics_collection.py
        └── visualization.py
```

- **`main.py`**: FastAPI entry point (with global exception handlers)  
- **`test_form.html`**: Testing interface for local & HPC endpoints  
- **`start.sh`**: Script for container-based or Render deployment  
- **`src/`** subdirectories hold domain logic (quantum_system, HPC, IonQ, memristor, etc.).

---

## Requirements

- **Python 3.8+** (3.10 recommended)
- **pip** or another package manager
- **QuTiP** (for quantum simulation steps)
- **FastAPI**, **Uvicorn** for the web server
- **Pydantic** for request/response models
- **NumPy** (plus HPC dependencies if needed)

### Python Libraries

| Library     | Purpose                                            |
|-------------|----------------------------------------------------|
| `fastapi`   | Web API framework                                  |
| `uvicorn`   | ASGI server for serving FastAPI                    |
| `qutip`     | Quantum simulation / state manipulation            |
| `numpy`     | Numerical computations                             |
| `pydantic`  | Request/response validation                        |
| `...`       | HPC, IonQ, memristor, or other domain dependencies |

---

## Installation

1. **Clone** or **download** this repository:

   ```bash
   git clone https://github.com/polyym/quantum_simulator-backend.git
   cd quantum_simulator-backend
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
   If you haven't generated `requirements.txt`, run:
   ```bash
   pip install fastapi uvicorn qutip numpy pydantic
   pip freeze > requirements.txt
   ```

4. **Update** `requirements.txt` when adding new dependencies:
   ```bash
   pip freeze > requirements.txt
   ```

---

## Running the Server

1. **Activate** your virtual environment if not already active.
2. **Start** with Uvicorn:

   ```bash
   uvicorn main:app --reload --port 8000
   ```
   - The app is at [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - Check [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) for "OK" status.

3. **Deploy**  
   - If using Render or Docker, use `start.sh`:
     ```bash
     #!/usr/bin/env bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```

---

## Usage

### 1. Testing Locally via `test_form.html`

- **Open** `test_form.html` in your browser.  
- Local endpoints: (e.g., `/qsystem/create_system`, `/ionq/drb`, `/memristor/run_circuit`)  
- HPC synergy endpoints: (e.g., `/hpc/submit_job`, `/qsystem/submit_distributed_simulation`, etc.)  
- **Fill** out forms, **click** "Submit." Results appear in `<pre>`.

### 2. cURL / Postman

- **Local HPC** job:
  ```bash
  curl -X POST -H "Content-Type: application/json" \
       -d '{
         "job_id":"job123",
         "qubit_count":10,
         "code_distance":3,
         "num_cycles":2,
         "cpu_cores":2,
         "gpu_cards":0,
         "memory_gb":4
       }' \
       http://127.0.0.1:8000/hpc/submit_job
  ```
- **Distributed Simulation** (Quantum System):
  ```bash
  curl -X POST -H "Content-Type: application/json" \
       -d '{
         "job_id":"dist_sim_1",
         "num_qubits":10,
         "code_distance":3,
         "num_cycles":2,
         "cpu_cores":2,
         "gpu_cards":0,
         "memory_gb":4
       }' \
       http://127.0.0.1:8000/qsystem/submit_distributed_simulation
  ```
- **IonQ HPC**:
  ```bash
  curl -X POST -H "Content-Type: application/json" \
       -d '{
         "job_id":"ionq_job_1",
         "benchmark_type":"drb",
         "qubits":[0,1],
         "depth":10,
         "p2q":0.25,
         "cpu_cores":2,
         "gpu_cards":0,
         "memory_gb":4
       }' \
       http://127.0.0.1:8000/ionq/submit_hpc_ionq
  ```
- **Surface Code Multi-Round QEC**:
  ```bash
  curl -X POST -H "Content-Type: application/json" \
       -d '{
         "job_id":"sc_qec_job1",
         "distance":5,
         "rounds":10,
         "cpu_cores":2,
         "gpu_cards":0,
         "memory_gb":4
       }' \
       http://127.0.0.1:8000/surface_code/run_multi_round_qec
  ```

---

## Endpoints

1. **Quantum System** (`/qsystem/...`)  
   - Local: `POST /qsystem/create_system`, `POST /qsystem/apply_operation`, `POST /qsystem/measure`  
   - HPC synergy: `POST /qsystem/submit_distributed_simulation`

2. **HPC** (`/hpc/...`)  
   - `POST /hpc/submit_job` for general HPC jobs  
   - `GET /hpc/job_status?job_id=...`  
   - `GET /hpc/list_jobs` (optional)  
   - `DELETE /hpc/cancel_job?job_id=...`

3. **IonQ** (`/ionq/...`)  
   - Local: `POST /ionq/drb`, `POST /ionq/application`  
   - HPC synergy: `POST /ionq/submit_hpc_ionq`

4. **Memristor** (`/memristor/...`)  
   - Local: `POST /memristor/run_circuit`  
   - HPC synergy: `POST /memristor/submit_hpc_memristor`

5. **Surface Code** (`/surface_code/...`)  
   - Local: `POST /surface_code/measure_stabilizers`, `POST /surface_code/decode_syndrome`  
   - HPC synergy: `POST /surface_code/run_multi_round_qec`

6. **Health Check**  
   - `GET /health` => Returns `"status":"OK"` plus app version.

---

## Troubleshooting

1. **HTTP 405 (Method Not Allowed)**  
   - Verify the correct HTTP method (POST vs GET vs DELETE).

2. **404 (Not Found)**  
   - Confirm you typed the router prefix (`/qsystem`, `/hpc`, etc.) properly.

3. **JSON Errors**  
   - Check that your request body is valid JSON. For HPC synergy, watch out for bracket/quote mistakes in `parameters` or circuit arrays.

4. **CORS Issues**  
   - The default in `main.py` is `allow_origins=["*"]` for local dev. Restrict to your front-end domain in production.

5. **Performance**  
   - Large HPC tasks (e.g., big IonQ DRB or surface code with distance=25) might be slow. Expand HPC nodes or reduce problem size.

6. **HPC Resource Failures**  
   - If HPC resource manager says “Insufficient HPC resources”, reduce CPU/GPU or memory requests, or update `ResourceManager(total_cores=..., total_gpus=..., total_memory_gb=...)` for your cluster specs.

---

---

**Enjoy** exploring HPC, IonQ, Memristor, and quantum system features with this modular FastAPI + QuTiP backend. For contributions or questions, open an issue or create a pull request.