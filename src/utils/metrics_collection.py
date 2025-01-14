# src/utils/metrics_collection.py

"""
Metrics collection utilities for quantum system performance tracking.
Focuses on gathering and analyzing performance metrics across different
components based on the research papers' methodologies.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics to track"""
    GATE_FIDELITY = "gate_fidelity"
    READOUT_FIDELITY = "readout_fidelity"
    COHERENCE_TIME = "coherence_time"
    CIRCUIT_DEPTH = "circuit_depth"
    EXECUTION_TIME = "execution_time"
    LOGICAL_ERROR_RATE = "logical_error_rate"
    PHYSICAL_ERROR_RATE = "physical_error_rate"
    MEMORY_LIFETIME = "memory_lifetime"

@dataclass
class MetricValue:
    """Container for a single metric measurement"""
    value: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    uncertainty: Optional[float] = None
    units: Optional[str] = None

class MetricsCollector:
    """Core metrics collection and analysis"""
    
    def __init__(self):
        self.metrics: Dict[MetricType, List[MetricValue]] = defaultdict(list)
        self.metric_thresholds: Dict[MetricType, float] = {}
        self.alert_callbacks: Dict[MetricType, List[callable]] = defaultdict(list)

    def record_metric(self, 
                     metric_type: MetricType,
                     value: float,
                     metadata: Optional[Dict[str, Any]] = None,
                     uncertainty: Optional[float] = None,
                     units: Optional[str] = None) -> None:
        """
        Record a new metric measurement
        
        Args:
            metric_type: Type of metric being recorded
            value: Measured value
            metadata: Additional contextual information
            uncertainty: Measurement uncertainty if applicable
            units: Units of measurement
        """
        try:
            metric = MetricValue(
                value=value,
                timestamp=datetime.now().timestamp(),
                metadata=metadata,
                uncertainty=uncertainty,
                units=units
            )
            self.metrics[metric_type].append(metric)
            
            # Check thresholds and trigger alerts if needed
            self._check_threshold(metric_type, value)
            
        except Exception as e:
            logger.error(f"Error recording metric: {str(e)}")

    def set_threshold(self, metric_type: MetricType, threshold: float,
                     callback: Optional[callable] = None) -> None:
        """Set threshold and optional callback for a metric type"""
        self.metric_thresholds[metric_type] = threshold
        if callback:
            self.alert_callbacks[metric_type].append(callback)

    def get_metric_stats(self, 
                        metric_type: MetricType,
                        window_size: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate statistics for a metric type
        
        Args:
            metric_type: Type of metric to analyze
            window_size: Optional, number of recent measurements to consider
            
        Returns:
            Dictionary of statistical measures
        """
        try:
            values = [m.value for m in self.metrics[metric_type]]
            if window_size:
                values = values[-window_size:]
                
            if not values:
                return {}
                
            return {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metric stats: {str(e)}")
            return {}

    def analyze_trend(self, 
                     metric_type: MetricType,
                     window_size: Optional[int] = None) -> Dict[str, float]:
        """
        Analyze trend for a metric type
        
        Args:
            metric_type: Type of metric to analyze
            window_size: Optional time window for analysis
            
        Returns:
            Dictionary containing trend analysis
        """
        try:
            metrics = self.metrics[metric_type]
            if window_size:
                metrics = metrics[-window_size:]
                
            if len(metrics) < 2:
                return {}
                
            values = [m.value for m in metrics]
            times = [m.timestamp for m in metrics]
            
            # Calculate linear regression
            coeffs = np.polyfit(times, values, 1)
            slope, intercept = coeffs
            
            # Calculate R-squared
            y_pred = np.polyval(coeffs, times)
            r_squared = 1 - (np.sum((np.array(values) - y_pred) ** 2) / 
                           np.sum((np.array(values) - np.mean(values)) ** 2))
                           
            return {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "direction": "increasing" if slope > 0 else "decreasing"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return {}

    def _check_threshold(self, metric_type: MetricType, value: float) -> None:
        """Check if a metric value crosses its threshold and trigger alerts"""
        if metric_type in self.metric_thresholds:
            threshold = self.metric_thresholds[metric_type]
            if value > threshold:
                for callback in self.alert_callbacks[metric_type]:
                    try:
                        callback(metric_type, value, threshold)
                    except Exception as e:
                        logger.error(f"Error in threshold callback: {str(e)}")

class PerformanceAnalyzer:
    """Analyze system performance metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector

    def analyze_system_performance(self) -> Dict[str, Any]:
        """
        Comprehensive system performance analysis
        
        Returns:
            Dictionary containing various performance metrics and analysis
        """
        try:
            return {
                "gate_performance": self._analyze_gate_metrics(),
                "memory_performance": self._analyze_memory_metrics(),
                "system_stability": self._analyze_stability_metrics()
            }
        except Exception as e:
            logger.error(f"Error analyzing system performance: {str(e)}")
            return {}

    def _analyze_gate_metrics(self) -> Dict[str, Any]:
        """Analyze gate-related metrics"""
        return {
            "fidelity": self.collector.get_metric_stats(MetricType.GATE_FIDELITY),
            "trend": self.collector.analyze_trend(MetricType.GATE_FIDELITY)
        }

    def _analyze_memory_metrics(self) -> Dict[str, Any]:
        """Analyze memory-related metrics"""
        return {
            "coherence": self.collector.get_metric_stats(MetricType.COHERENCE_TIME),
            "lifetime": self.collector.get_metric_stats(MetricType.MEMORY_LIFETIME)
        }

    def _analyze_stability_metrics(self) -> Dict[str, Any]:
        """Analyze system stability metrics"""
        return {
            "error_rates": {
                "logical": self.collector.get_metric_stats(MetricType.LOGICAL_ERROR_RATE),
                "physical": self.collector.get_metric_stats(MetricType.PHYSICAL_ERROR_RATE)
            },
            "readout_fidelity": self.collector.get_metric_stats(MetricType.READOUT_FIDELITY)
        }