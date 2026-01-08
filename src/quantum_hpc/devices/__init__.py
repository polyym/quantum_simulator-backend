# src/quantum_hpc/devices/__init__.py

"""
Devices Package

QEC code implementations and device-specific models:
- surface_code: Surface code error correction

Version: 3.0.0
"""

from . import surface_code

__all__ = [
    "surface_code",
]
