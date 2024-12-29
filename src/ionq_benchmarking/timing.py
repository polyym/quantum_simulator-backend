# src/ionq_benchmarking/timing.py

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging

@dataclass
class GateTiming:
    """Gate timing information from paper"""
    single_qubit: float = 110.0  # 110 microseconds for single-qubit gate
    zz_gate: float = 900.0       # ~900 microseconds average for ZZ gate
    state_prep: float = 0.5      # State preparation time in milliseconds
    measurement: float = 0.5     # Measurement time in milliseconds
    aod_switch: float = 0.1      # AOD switching time in milliseconds

@dataclass
class CircuitTiming:
    """Circuit execution timing information"""
    total_time: float = 0.0      # Total execution time
    gate_time: float = 0.0       # Time spent on gates
    overhead_time: float = 0.0   # Time spent on overhead operations
    shots: int = 100             # Number of shots executed

class TimingAnalyzer:
    """Analyze and track timing performance of quantum circuits"""
    def __init__(self):
        self.gate_timing = GateTiming()
        self.circuit_timings = {}
        self.start_time = None
        self.current_circuit = None
        
    def start_circuit(self, circuit_id: str):
        """Start timing a new circuit"""
        self.start_time = datetime.now()
        self.current_circuit = circuit_id
        
    def end_circuit(self, shots: int = 100):
        """End timing for current circuit"""
        if self.start_time and self.current_circuit:
            execution_time = (datetime.now() - self.start_time).total_seconds() * 1000  # Convert to ms
            self.circuit_timings[self.current_circuit] = CircuitTiming(
                total_time=execution_time,
                shots=shots
            )
            self.start_time = None
            self.current_circuit = None

    def analyze_circuit_timing(self, 
                             circuit: List[Dict],
                             include_overhead: bool = True) -> Dict[str, float]:
        """
        Analyze timing of a quantum circuit based on IonQ Forte specifications
        Args:
            circuit: List of quantum operations
            include_overhead: Whether to include state prep and measurement overhead
        Returns:
            Dictionary containing timing analysis
        """
        try:
            # Initialize timing components
            gate_time = 0.0
            switching_time = 0.0
            overhead_time = 0.0
            
            # Track previous qubits for AOD switching
            prev_qubits = set()
            
            # Analyze each operation
            for op in circuit:
                # Gate time
                if op['type'] in ['H', 'X', 'Y', 'Z']:  # Single-qubit gates
                    gate_time += self.gate_timing.single_qubit
                elif op['type'] == 'ZZ':  # Two-qubit ZZ gate
                    gate_time += self.gate_timing.zz_gate
                    
                # AOD switching overhead
                current_qubits = set(op['qubits'])
                if current_qubits != prev_qubits:
                    switching_time += self.gate_timing.aod_switch
                prev_qubits = current_qubits
            
            # Add state preparation and measurement if requested
            if include_overhead:
                overhead_time = self.gate_timing.state_prep + self.gate_timing.measurement
            
            total_time = gate_time + switching_time + overhead_time
            
            return {
                'total_time_ms': total_time,
                'gate_time_ms': gate_time,
                'switching_time_ms': switching_time,
                'overhead_time_ms': overhead_time,
                'gate_percentage': (gate_time / total_time) * 100 if total_time > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Error in timing analysis: {str(e)}")
            return {}

class ApplicationTimingTracker:
    """Track timing for specific quantum applications from the paper"""
    def __init__(self):
        self.analyzer = TimingAnalyzer()
        self.application_timings = {}
        
    def track_application(self, 
                         name: str, 
                         circuit_width: int, 
                         num_gates: int,
                         shots: int = 100):
        """
        Track timing for a specific application benchmark
        Args:
            name: Application name
            circuit_width: Number of qubits
            num_gates: Number of gates in circuit
            shots: Number of shots executed
        """
        try:
            self.analyzer.start_circuit(f"{name}_{circuit_width}")
            
            # Calculate estimated execution time based on paper's model
            gate_time = (
                num_gates * (
                    0.7 * self.analyzer.gate_timing.single_qubit +  # 70% single-qubit gates
                    0.3 * self.analyzer.gate_timing.zz_gate        # 30% two-qubit gates
                )
            )
            
            # Add overhead
            total_time = (
                gate_time +
                self.analyzer.gate_timing.state_prep +
                self.analyzer.gate_timing.measurement +
                (num_gates * self.analyzer.gate_timing.aod_switch * 0.1)  # Estimated switching overhead
            )
            
            # Store timing information
            self.application_timings[f"{name}_{circuit_width}"] = {
                'circuit_width': circuit_width,
                'num_gates': num_gates,
                'shots': shots,
                'estimated_time_ms': total_time,
                'gate_time_percentage': (gate_time / total_time) * 100 if total_time > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Error tracking application timing: {str(e)}")

    def get_application_statistics(self) -> Dict:
        """Get statistics about application timing performance"""
        if not self.application_timings:
            return {}
            
        times = [timing['estimated_time_ms'] 
                for timing in self.application_timings.values()]
        
        return {
            'average_time_ms': np.mean(times),
            'max_time_ms': np.max(times),
            'min_time_ms': np.min(times),
            'std_dev_ms': np.std(times),
            'total_applications': len(self.application_timings)
        }

    def get_timing_by_width(self) -> Dict[int, List[float]]:
        """Get timing statistics grouped by circuit width"""
        width_timings = {}
        
        for app_name, timing in self.application_timings.items():
            width = timing['circuit_width']
            if width not in width_timings:
                width_timings[width] = []
            width_timings[width].append(timing['estimated_time_ms'])
            
        return {
            width: {
                'average_time_ms': np.mean(times),
                'std_dev_ms': np.std(times),
                'num_circuits': len(times)
            }
            for width, times in width_timings.items()
        }