"""
main.py

Entrypoint for the quantum simulator backend, defining the FastAPI app
and mounting each router for different quantum domains (Quantum System,
HPC, IonQ, Memristor-based gates, etc.).
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

# Import routers (files from src/routers/):
from src.routers.quantum_system_router import router as quantum_system_router
from src.routers.hpc_router import router as hpc_router
from src.routers.ionq_router import router as ionq_router
from src.routers.memristor_router import router as memristor_router
from src.routers.surface_code_router import router as surface_code_router

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
        "with separate modules for HPC scheduling, IonQ benchmarking, "
        "memristor-based gates, and general quantum system functionalities."
    ),
    version="2.0.0",
)

# -------------------------------------------------------------------------
# CORS Middleware
# NOTE: For production, restrict allow_origins to known front-end URLs.
# -------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# Global Exception Handlers (Optional, but recommended)
# -------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Returns a uniform JSON structure for HTTP exceptions (e.g., 404, 400).
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail
        },
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """
    Returns a uniform JSON structure for Pydantic validation errors.
    """
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "detail": exc.errors()
        },
    )

# -------------------------------------------------------------------------
# Include Routers
# -------------------------------------------------------------------------
app.include_router(quantum_system_router, prefix="/qsystem", tags=["QuantumSystem"])
app.include_router(hpc_router, prefix="/hpc", tags=["HPC"])
app.include_router(ionq_router, prefix="/ionq", tags=["IonQ"])
app.include_router(memristor_router, prefix="/memristor", tags=["Memristor"])
app.include_router(surface_code_router, prefix="/surface_code", tags=["SurfaceCode"])

# -------------------------------------------------------------------------
# Simple Health Check
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

# -------------------------------------------------------------------------
# Optional Uvicorn Entry Point
# -------------------------------------------------------------------------
# If you want to run this file directly (e.g., `python main.py`),
# uncomment the block below. Otherwise, you can run with:
#   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# in your CLI or Dockerfile.
# -------------------------------------------------------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )
