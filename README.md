Below is an **updated** `README.md` that **includes** all the essential steps regarding:

- Creating a **virtual environment**  
- Installing/updating **requirements**  
- Using **`test_form.html`** to test the API in a **local** environment  

Feel free to tweak any details (like repository name or command specifics) to match your actual setup.

---

```markdown
# Quantum Simulator Backend

This project contains a **FastAPI + QuTiP** backend designed to handle a variety of quantum computing functionalities:

- **Quantum System Simulation** (general gates, noise models, measurement)
- **HPC** (High-Performance Computing) quantum system creation/management
- **IonQ**-style benchmarking (DRB, application benchmarks, etc.)
- **Memristor-based** quantum gate accelerators

All features are exposed via **HTTP endpoints** for easy integration with frontends or external clients. A **router-based design** ensures each domain (HPC, IonQ, Memristor, and general quantum system) remains modular.

---

## Table of Contents

1. [Features](#features)  
2. [Repository Structure](#repository-structure)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Running the Server](#running-the-server)  
6. [Usage](#usage)  
7. [Endpoints](#endpoints)  
8. [Troubleshooting](#troubleshooting)  
9. [License](#license)

---

## Features

- **Quantum System**  
  - Apply single- and multi-qubit gates (H, X, Y, Z, Phase, T, Swap, CNOT, etc.)  
  - Noise modeling (amplitude, phase damping, depolarizing)  
  - Retrieve final quantum state as `[real, imag]` pairs  
  - Perform measurements and retrieve shot counts  

- **HPC**  
  - Abstract machine models (Asymmetric, Accelerator, Quantum Accelerator)  
  - HPC system creation, HPC instructions, distributed algorithms  

- **IonQ**  
  - IonQ device creation (`IonQDevice`)  
  - Direct Randomized Benchmarking (DRB), application benchmarks (hamiltonian_simulation, phase_estimation, QFT, etc.)  
  - Error mitigation, circuit optimization, timing analysis  

- **Memristor Gates**  
  - Parallel quantum operations on a memristor-based crossbar  
  - Detailed power metrics (static/dynamic power, switching energy)  
  - Additional gates: SWAP, √SWAP, CCNOT, etc.

- **Router-Based Design**  
  - Separate routers for HPC, IonQ, Memristor, and general quantum system  
  - `main.py` orchestrates and includes each router with a prefix (e.g. `/hpc`, `/ionq`, `/qsystem`, `/memristor`)

---

## Repository Structure

```
quantum_simulator-backend/
├── main.py
├── requirements.txt
├── README.md
├── test_form.html
└── src/
    ├── routers/
    │   ├── quantum_system_router.py
    │   ├── hpc_router.py
    │   ├── ionq_router.py
    │   └── memristor_router.py
    ├── quantum_system/
    ├── quantum_hpc/
    ├── ionq_benchmarking/
    └── memristor_gates/
```

- **`main.py`**: FastAPI entry point. Imports each router and includes them.  
- **`src/routers/`**: Each router file (HPC, IonQ, Memristor, quantum system, etc.) has domain-specific endpoints.  
- **`test_form.html`**: A simple HTML file to test endpoints locally in your browser.  

---

## Requirements

- **Python 3.8+** (3.10 or 3.11 recommended)  
- **pip** or another package manager  
- **QuTiP** for quantum simulation  
- **FastAPI** and **Uvicorn** for the web server

### Python Libraries

| Library   | Purpose                                   |
|-----------|-------------------------------------------|
| `fastapi` | Web API framework                         |
| `uvicorn` | ASGI server to run FastAPI                |
| `qutip`   | Quantum simulation & operations           |
| `numpy`   | Numerical computations                    |
| `pydantic`| Validation of request payloads            |
| `...`     | HPC, IonQ, memristor, or other deps       |

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
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - If you don’t have a `requirements.txt` file yet, run:
     ```bash
     pip install fastapi uvicorn qutip
     OR
     python -m pip install fastapi uvicorn qutip
     ```
     Then:
     ```bash
     pip freeze > requirements.txt
     ```
   This ensures all needed packages (FastAPI, Uvicorn, QuTiP, etc.) are installed.

4. **(Optional) Update the requirements** if you add new dependencies:
   ```bash
   pip freeze > requirements.txt
   ```

---

## Running the Server

1. **Activate** your virtual environment (if not already).
2. **Run** the FastAPI server using Uvicorn:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   - `--reload` auto-restarts on code changes.
   - Open <http://127.0.0.1:8000/health> in your browser to confirm a basic “OK” response.

---

## Usage

### 1. Test Locally with `test_form.html`

- In this repo, there’s a file **`test_form.html`** at the root.
- **Open** `test_form.html` in a **browser** (double-click or drag it into a tab).
- You’ll see multiple sections for HPC, IonQ, quantum system `/simulate`, and memristor gates.
- Enter the desired inputs (number of qubits, HPC model, gate definitions, etc.) and click the **button** to send a **fetch** request to the corresponding endpoints.
- The **response** appears in a `<pre>` block at the bottom.

### 2. Using cURL or Postman

You can also **POST** or **GET** calls via cURL, Postman, or other REST clients. For example:

- **Quantum System**:  
  ```bash
  curl -X POST -H "Content-Type: application/json" \
       -d '{
         "num_qubits": 2,
         "gates": [
           {"gate": "H", "qubits": [0]},
           {"gate": "CNOT", "qubits": [0,1]}
         ],
         "shots": 1000
       }' \
       http://127.0.0.1:8000/qsystem/simulate
  ```
- **HPC**:  
  ```bash
  curl -X POST -H "Content-Type: application/json" \
       -d '{"num_nodes":3,"model_type":"asymmetric"}' \
       http://127.0.0.1:8000/hpc/create
  ```
- **IonQ**:  
  ```bash
  curl -X POST -H "Content-Type: application/json" \
       -d '{"device_id":"ionq-123","qubits":[0,1],"depth":10,"p2q":0.25}' \
       http://127.0.0.1:8000/ionq/device/drb
  ```
- **Memristor**:  
  ```bash
  curl -X POST -H "Content-Type: application/json" \
       -d '{"max_parallel_ops":4}' \
       http://127.0.0.1:8000/memristor/accelerator/create
  ```

---

## Endpoints

Since each feature is in its own router:

- **HPC** (`/hpc/...`)  
  - `POST /hpc/create`  
  - `GET /hpc/{hpc_id}/info`  
  - etc.

- **IonQ** (`/ionq/...`)  
  - `POST /ionq/device/create`  
  - `POST /ionq/device/drb`  
  - etc.

- **Quantum System** (`/qsystem/...`)  
  - `POST /qsystem/simulate`  
  - etc.

- **Memristor** (`/memristor/...`)  
  - `POST /memristor/accelerator/create`  
  - `POST /memristor/accelerator/execute_circuit`  
  - etc.

Additionally, you might have a `GET /health` (in `main.py` or one of the routers) returning basic status info.

---

## Troubleshooting

1. **Method Not Allowed (405)**  
   - Check if you’re using the **correct** HTTP method (e.g., POST instead of GET).
   - Verify the **URL** includes the correct router prefix.

2. **JSON Serialization Issues**  
   - Complex numbers are returned as lists of `[real, imag]`. If you see `"complex" object is not JSON serializable`, confirm your code does the conversion to `[real, imag]` pairs.

3. **Large Qubit Counts**  
   - QuTiP can get slow with large `num_qubits` because the state-space grows exponentially. Try smaller circuits or partial solutions for big HPC tasks.

4. **CORS or Browser Errors**  
   - By default, `allow_origins=["*"]` is set for development in `main.py`. For production, restrict to specific domains.

5. **Missing Endpoints**  
   - If 404, confirm `main.py` calls `app.include_router(...)` for the desired router, and that the **prefix** matches your request path.

---

## License

(Include your license of choice, e.g., MIT, Apache 2.0, or proprietary.)

---

**Enjoy** exploring HPC, IonQ, Memristor, and quantum system features with this modular FastAPI + QuTiP backend. For contributions or questions, open an issue or create a pull request.