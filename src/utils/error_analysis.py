# src/utils/error_analysis.py

"""
Utilities for analyzing quantum errors and error rates, supporting both
physical and logical error analysis based on the Google surface code paper
and IonQ benchmarking approaches.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Types of quantum errors we track"""
    PHYSICAL = "physical"
    LOGICAL = "logical"
    MEASUREMENT = "measurement"
    GATE = "gate"
    DECOHERENCE = "decoherence"
    LEAKAGE = "leakage"
    CROSS_TALK = "cross_talk"

@dataclass
class ErrorMetrics:
    """Metrics for error tracking"""
    error_rate: float
    error_type: ErrorType
    confidence_interval: Optional[Tuple[float, float]] = None
    num_events: int = 0
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ErrorAnalyzer:
    """Core error analysis functionality"""
    
    def __init__(self):
        self.error_history: Dict[ErrorType, List[ErrorMetrics]] = {
            error_type: [] for error_type in ErrorType
        }
        self.threshold_cache: Dict[ErrorType, Tuple[float, float]] = {}

    def record_error(self, error_rate: float, error_type: ErrorType, 
                    num_events: int = 1, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a new error measurement
        
        Args:
            error_rate: Measured error rate
            error_type: Type of error being recorded
            num_events: Number of error events
            metadata: Additional information about the error
        """
        try:
            metrics = ErrorMetrics(
                error_rate=error_rate,
                error_type=error_type,
                num_events=num_events,
                timestamp=datetime.now().timestamp(),
                metadata=metadata
            )
            self.error_history[error_type].append(metrics)
            # Invalidate threshold cache for this error type
            self.threshold_cache.pop(error_type, None)
            
        except Exception as e:
            logger.error(f"Error recording error metrics: {str(e)}")

    def calculate_error_threshold(self, error_type: ErrorType,
                                confidence_level: float = 0.95,
                                window_size: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate error threshold with confidence interval
        
        Args:
            error_type: Type of error to analyze
            confidence_level: Confidence level for interval calculation
            window_size: Optional, number of recent measurements to consider
            
        Returns:
            Tuple of (threshold, margin)
        """
        if not self.error_history[error_type]:
            return (0.0, 0.0)
            
        try:
            # Use cached value if available and window_size is None
            if window_size is None and error_type in self.threshold_cache:
                return self.threshold_cache[error_type]
                
            metrics = self.error_history[error_type]
            if window_size:
                metrics = metrics[-window_size:]
                
            rates = [m.error_rate for m in metrics]
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)
            
            # Calculate confidence interval
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_score * (std_rate / np.sqrt(len(rates)))
            
            result = (mean_rate, margin)
            
            # Cache result if using full history
            if window_size is None:
                self.threshold_cache[error_type] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error calculating threshold: {str(e)}")
            return (0.0, 0.0)

    def analyze_error_scaling(self, base_distance: int,
                            increased_distance: int,
                            physical_error_rate: float) -> Dict[str, float]:
        """
        Analyze how errors scale with code distance, based on the Google paper's
        approach to surface code scaling
        
        Args:
            base_distance: Initial code distance
            increased_distance: Larger code distance to compare against
            physical_error_rate: Underlying physical error rate
            
        Returns:
            Dictionary containing scaling metrics
        """
        try:
            # Implementation based on Google's surface code paper
            base_logical_rate = self.analyze_logical_error_rate(
                physical_error_rate, base_distance
            )
            increased_logical_rate = self.analyze_logical_error_rate(
                physical_error_rate, increased_distance
            )
            
            scaling_factor = base_logical_rate / increased_logical_rate
            
            return {
                "base_logical_rate": base_logical_rate,
                "increased_logical_rate": increased_logical_rate,
                "scaling_factor": scaling_factor,
                "threshold_estimate": self._estimate_threshold(
                    base_distance, increased_distance,
                    base_logical_rate, increased_logical_rate
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing scaling: {str(e)}")
            return {}

    def analyze_logical_error_rate(self, physical_error_rate: float,
                                 code_distance: int) -> float:
        """
        Calculate expected logical error rate based on physical error rate
        and code distance using the threshold theorem
        
        Args:
            physical_error_rate: Underlying physical error rate
            code_distance: Distance of the error correction code
            
        Returns:
            Expected logical error rate
        """
        try:
            # Based on the threshold theorem and surface code scaling
            threshold = 1e-2  # Example threshold from papers
            
            if physical_error_rate >= threshold:
                return 1.0  # Above threshold, no improvement
                
            # Calculate logical error rate based on code distance
            # Using simplified model from papers
            suppression_factor = (physical_error_rate / threshold) ** ((code_distance + 1) / 2)
            logical_error_rate = threshold * suppression_factor
            
            return logical_error_rate
            
        except Exception as e:
            logger.error(f"Error calculating logical error rate: {str(e)}")
            return 1.0

    def _estimate_threshold(self, d1: int, d2: int, 
                          p1: float, p2: float) -> float:
        """
        Estimate the error threshold using two different distances
        
        Args:
            d1, d2: Code distances
            p1, p2: Corresponding logical error rates
            
        Returns:
            Estimated threshold
        """
        try:
            # Based on the crossing of logical error rate curves
            # from the surface code papers
            if d1 >= d2 or p1 <= p2:
                return 0.0
                
            # Calculate intersection point
            alpha = np.log(p2/p1) / (d2 - d1)
            threshold = np.exp(-alpha)
            
            return threshold
            
        except Exception as e:
            logger.error(f"Error estimating threshold: {str(e)}")
            return 0.0

class ErrorCorrelationAnalyzer:
    """Analyze correlations between different types of errors"""
    
    def __init__(self):
        self.correlation_data: Dict[Tuple[ErrorType, ErrorType], List[float]] = {}
        
    def add_correlation_data(self, type1: ErrorType, type2: ErrorType, 
                           correlation: float) -> None:
        """Add a correlation measurement between two error types"""
        key = tuple(sorted([type1, type2]))
        if key not in self.correlation_data:
            self.correlation_data[key] = []
        self.correlation_data[key].append(correlation)

    def analyze_error_correlations(self, 
                                 recent_only: bool = False,
                                 window_size: int = 100) -> Dict[Tuple[ErrorType, ErrorType], float]:
        """
        Analyze correlations between different error types
        
        Args:
            recent_only: If True, only analyze recent correlations
            window_size: Number of recent measurements to consider if recent_only=True
            
        Returns:
            Dictionary mapping error type pairs to their correlation strength
        """
        try:
            results = {}
            for (type1, type2), correlations in self.correlation_data.items():
                if recent_only:
                    correlations = correlations[-window_size:]
                if correlations:
                    results[(type1, type2)] = np.mean(correlations)
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            return {}

def calculate_confidence_interval(data: List[float],
                                confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for error measurements
    
    Args:
        data: List of measurements
        confidence_level: Desired confidence level
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    try:
        if not data:
            return (0.0, 0.0)
            
        mean = np.mean(data)
        std = np.std(data)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * (std / np.sqrt(len(data)))
        
        return (mean - margin, mean + margin)
        
    except Exception as e:
        logger.error(f"Error calculating confidence interval: {str(e)}")
        return (0.0, 0.0)