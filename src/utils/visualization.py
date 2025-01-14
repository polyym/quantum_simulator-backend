# src/utils/visualization.py

"""
Visualization utilities for quantum system analysis.
Provides tools to generate various visualizations of quantum system performance,
error rates, and other metrics. Designed to work with metrics_collection.py
and error_analysis.py data.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass

# Import from our other utility modules
from .metrics_collection import MetricType, MetricValue
from .error_analysis import ErrorType, ErrorMetrics

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization outputs"""
    title: str
    x_label: str
    y_label: str
    color_scheme: Optional[str] = "blues"
    show_error_bars: bool = True
    show_trend_line: bool = True
    time_window: Optional[int] = None

class DataFormatter:
    """Format data for visualization"""
    
    @staticmethod
    def format_time_series(metrics: List[MetricValue]) -> Tuple[List[float], List[float]]:
        """Convert metric values into time series format"""
        try:
            times = [m.timestamp for m in metrics]
            values = [m.value for m in metrics]
            return times, values
        except Exception as e:
            logger.error(f"Error formatting time series: {str(e)}")
            return [], []

    @staticmethod
    def format_error_data(error_metrics: List[ErrorMetrics]) -> Dict[str, List[float]]:
        """Format error metrics for visualization"""
        try:
            return {
                "rates": [m.error_rate for m in error_metrics],
                "times": [m.timestamp for m in error_metrics],
                "counts": [m.num_events for m in error_metrics]
            }
        except Exception as e:
            logger.error(f"Error formatting error data: {str(e)}")
            return {"rates": [], "times": [], "counts": []}

class QuantumStateVisualizer:
    """Visualize quantum states and operations"""
    
    @staticmethod
    def generate_state_visualization(state_vector: List[complex]) -> Dict[str, Any]:
        """
        Generate visualization data for a quantum state
        
        Args:
            state_vector: Complex amplitudes of quantum state
            
        Returns:
            Dictionary containing visualization data
        """
        try:
            amplitudes = np.array(state_vector)
            probabilities = np.abs(amplitudes) ** 2
            phases = np.angle(amplitudes)
            
            return {
                "probabilities": probabilities.tolist(),
                "phases": phases.tolist(),
                "basis_states": [format(i, f'0{int(np.log2(len(state_vector)))}b') 
                               for i in range(len(state_vector))]
            }
        except Exception as e:
            logger.error(f"Error generating state visualization: {str(e)}")
            return {}

    @staticmethod
    def generate_density_matrix_visualization(density_matrix: np.ndarray) -> Dict[str, Any]:
        """Generate visualization data for a density matrix"""
        try:
            real_part = np.real(density_matrix)
            imag_part = np.imag(density_matrix)
            
            return {
                "real": real_part.tolist(),
                "imaginary": imag_part.tolist(),
                "magnitude": np.abs(density_matrix).tolist()
            }
        except Exception as e:
            logger.error(f"Error generating density matrix visualization: {str(e)}")
            return {}

class ErrorVisualizer:
    """Visualize error rates and patterns"""
    
    @staticmethod
    def generate_error_heatmap(error_rates: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
        """
        Generate heatmap data for error rates between qubit pairs
        Based on Google's surface code visualization approach
        """
        try:
            # Convert to matrix format
            max_idx = max(max(i, j) for i, j in error_rates.keys()) + 1
            matrix = np.zeros((max_idx, max_idx))
            
            for (i, j), rate in error_rates.items():
                matrix[i, j] = rate
                matrix[j, i] = rate  # Assume symmetry
                
            return {
                "matrix": matrix.tolist(),
                "max_value": float(np.max(matrix)),
                "min_value": float(np.min(matrix[matrix > 0]))
            }
        except Exception as e:
            logger.error(f"Error generating error heatmap: {str(e)}")
            return {}

    @staticmethod
    def generate_error_history_plot(error_metrics: List[ErrorMetrics]) -> Dict[str, Any]:
        """Generate time series plot of error rates"""
        try:
            formatted = DataFormatter.format_error_data(error_metrics)
            
            return {
                "times": formatted["times"],
                "rates": formatted["rates"],
                "trend": np.polyfit(formatted["times"], formatted["rates"], 1).tolist()
                if len(formatted["times"]) > 1 else []
            }
        except Exception as e:
            logger.error(f"Error generating error history plot: {str(e)}")
            return {}

class PerformanceVisualizer:
    """Visualize system performance metrics"""
    
    @staticmethod
    def generate_metric_dashboard(
        metrics: Dict[MetricType, List[MetricValue]],
        config: Optional[VisualizationConfig] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard data"""
        try:
            dashboard = {}
            for metric_type, values in metrics.items():
                times, metric_values = DataFormatter.format_time_series(values)
                
                dashboard[metric_type.value] = {
                    "times": times,
                    "values": metric_values,
                    "stats": {
                        "mean": float(np.mean(metric_values)) if metric_values else 0,
                        "std": float(np.std(metric_values)) if metric_values else 0,
                        "trend": np.polyfit(times, metric_values, 1).tolist()
                        if len(times) > 1 else []
                    }
                }
            return dashboard
        except Exception as e:
            logger.error(f"Error generating metric dashboard: {str(e)}")
            return {}

    @staticmethod
    def generate_stability_plot(
        coherence_times: List[float],
        gate_fidelities: List[float],
        timestamps: List[float]
    ) -> Dict[str, Any]:
        """Generate stability analysis visualization"""
        try:
            return {
                "times": timestamps,
                "coherence": {
                    "values": coherence_times,
                    "trend": np.polyfit(timestamps, coherence_times, 1).tolist()
                    if len(timestamps) > 1 else []
                },
                "fidelity": {
                    "values": gate_fidelities,
                    "trend": np.polyfit(timestamps, gate_fidelities, 1).tolist()
                    if len(timestamps) > 1 else []
                }
            }
        except Exception as e:
            logger.error(f"Error generating stability plot: {str(e)}")
            return {}

def format_visualization_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format any numpy types to Python native types for JSON serialization
    
    Args:
        data: Dictionary containing visualization data
        
    Returns:
        Dictionary with all values converted to JSON-serializable types
    """
    try:
        def convert_value(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, (np.int_, np.int32, np.int64)):
                return int(v)
            elif isinstance(v, (np.float_, np.float32, np.float64)):
                return float(v)
            elif isinstance(v, dict):
                return {k: convert_value(v) for k, v in v.items()}
            elif isinstance(v, list):
                return [convert_value(i) for i in v]
            return v

        return {k: convert_value(v) for k, v in data.items()}
        
    except Exception as e:
        logger.error(f"Error formatting visualization data: {str(e)}")
        return {}