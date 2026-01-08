"""
main.py

Entrypoint for the quantum simulator backend, defining the FastAPI app
and mounting each router for different quantum domains (Quantum System,
HPC, IonQ, Memristor-based gates, etc.).
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT as HTTP_422_UNPROCESSABLE_ENTITY

# Configuration
from src.config import get_settings

# Shared services
from src.services import cleanup_services, get_hpc_coordinator, get_quantum_system_manager

# Import routers (files from src/routers/):
from src.routers.quantum_system_router import router as quantum_system_router
from src.routers.hpc_router import router as hpc_router
from src.routers.ionq_router import router as ionq_router
from src.routers.memristor_router import router as memristor_router
from src.routers.surface_code_router import router as surface_code_router

# -------------------------------------------------------------------------
# Settings and Logging Configuration
# -------------------------------------------------------------------------
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Application Lifespan Management
# -------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.
    Initializes shared services on startup and cleans them up on shutdown.
    """
    # Startup: Initialize shared services
    logger.info("Starting up Quantum Circuit Simulator...")
    logger.info("Initializing HPC Coordinator...")
    get_hpc_coordinator()
    logger.info("Initializing Quantum System Manager...")
    get_quantum_system_manager()
    logger.info("All services initialized successfully")

    yield

    # Shutdown: Cleanup shared services
    logger.info("Shutting down Quantum Circuit Simulator...")
    cleanup_services()
    logger.info("Shutdown complete")


# -------------------------------------------------------------------------
# FastAPI Application
# -------------------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    description=(
        "A production-ready FastAPI backend for quantum circuit simulation using QuTiP, "
        "featuring HPC job coordination, IonQ-style benchmarking, memristor gate acceleration, "
        "and surface code error correction."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

# -------------------------------------------------------------------------
# CORS Middleware
# -------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

logger.info(f"CORS configured with origins: {settings.cors_origins}")


# -------------------------------------------------------------------------
# Global Exception Handlers
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Returns a uniform JSON structure for request validation errors.
    """
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "detail": exc.errors()
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected exceptions.
    Logs the error and returns a generic error response.
    """
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "An unexpected error occurred. Please try again later."
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
# Health Check Endpoint
# -------------------------------------------------------------------------
@app.get("/health")
async def root_health():
    """
    Health check endpoint that verifies core services are operational.
    """
    try:
        # Verify HPC service is available
        hpc = get_hpc_coordinator()
        hpc_status = "healthy" if hpc is not None else "unavailable"

        # Verify quantum system manager is available
        qsm = get_quantum_system_manager()
        qsm_status = "healthy" if qsm is not None else "unavailable"
        active_systems = qsm.count() if qsm else 0

        return {
            "status": "OK",
            "version": app.version,
            "services": {
                "hpc_coordinator": hpc_status,
                "quantum_system_manager": qsm_status,
            },
            "metrics": {
                "active_quantum_systems": active_systems,
                "active_hpc_jobs": len(hpc.jobs) if hpc else 0,
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "version": app.version,
                "error": str(e)
            }
        )


# -------------------------------------------------------------------------
# Optional Uvicorn Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
