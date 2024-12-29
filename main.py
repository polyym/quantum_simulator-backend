"""
main.py

Entrypoint for the quantum simulator backend, defining the FastAPI app
and mounting each router for different quantum domains (Quantum System,
HPC, IonQ, Memristor-based gates, etc.).
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your routers here:
# Each of these files should exist in "src/routers/" and export a "router" object.
from src.routers.quantum_system_router import router as quantum_system_router
from src.routers.hpc_router import router as hpc_router
from src.routers.ionq_router import router as ionq_router
from src.routers.memristor_router import router as memristor_router

# -------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# FastAPI Application
# -------------------------------------------------------------------------
app = FastAPI(
    title="Quantum Circuit Simulator",
    description=(
        "A FastAPI backend for quantum circuit simulation using QuTiP, "
        "with separate modules for HPC, IonQ benchmarking, Memristor gates, "
        "and general quantum system functionalities."
    ),
    version="2.0.0",
)

# -------------------------------------------------------------------------
# CORS Middleware
# (In production, update 'allow_origins' for security)
# -------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# Include Routers
# -------------------------------------------------------------------------
app.include_router(quantum_system_router, prefix="/qsystem", tags=["QuantumSystem"])
app.include_router(hpc_router, prefix="/hpc", tags=["HPC"])
app.include_router(ionq_router, prefix="/ionq", tags=["IonQ"])
app.include_router(memristor_router, prefix="/memristor", tags=["Memristor"])

# -------------------------------------------------------------------------
# Simple Health Check (Optional)
# -------------------------------------------------------------------------
@app.get("/health")
async def root_health():
    """
    Basic health check for the API.
    """
    return {
        "status": "OK",
        "version": app.version,
    }
