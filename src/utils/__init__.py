# src/utils/__init__.py

"""
Utilities Package

Shared utility modules for quantum simulation:
- benchmarking: Performance evaluation and benchmark runners
- error_analysis: Quantum error analysis and statistics
- metrics_collection: Metrics recording and aggregation
- visualization: Data visualization helpers

Version: 3.0.0
"""

from .benchmarking import (
    BenchmarkType,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkManager,
)
from .error_analysis import ErrorType, ErrorAnalyzer
from .metrics_collection import MetricType, MetricsCollector

__all__ = [
    "BenchmarkType",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkManager",
    "ErrorType",
    "ErrorAnalyzer",
    "MetricType",
    "MetricsCollector",
]
