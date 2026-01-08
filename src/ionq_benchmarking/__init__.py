# src/ionq_benchmarking/__init__.py

"""
IonQ Benchmarking Package

IonQ-style benchmarking and error mitigation:
- Direct Randomized Benchmarking (DRB)
- Application benchmarks (Hamiltonian simulation, QFT)
- Error mitigation with circuit variants
- Timing analysis

Version: 3.0.0
"""

from .core import (
    IonQDevice,
    BenchmarkMetrics,
    ApplicationBenchmarks,
    BenchmarkType,
    GateMetrics,
)
from .error_mitigation import ErrorMitigation, CircuitOptimizer
from .timing import (
    TimingAnalyzer,
    ApplicationTimingTracker,
    GateTiming,
    CircuitTiming,
)

__all__ = [
    # Core
    "IonQDevice",
    "BenchmarkMetrics",
    "ApplicationBenchmarks",
    "BenchmarkType",
    "GateMetrics",
    # Error Mitigation
    "ErrorMitigation",
    "CircuitOptimizer",
    # Timing
    "TimingAnalyzer",
    "ApplicationTimingTracker",
    "GateTiming",
    "CircuitTiming",
]
