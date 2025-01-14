# src/utils/benchmarking.py

"""
Benchmarking utilities for quantum system performance evaluation.
Implements comprehensive benchmarking tools based on approaches from
Google's surface code paper and IonQ's benchmarking methodologies.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
from collections import defaultdict

# Import from our other utility modules
from .metrics_collection import MetricType, MetricsCollector
from .error_analysis import ErrorType, ErrorAnalyzer

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of quantum benchmarks"""
    DIRECT_RANDOMIZED = "direct_randomized"
    VOLUMETRIC = "volumetric"
    APPLICATION = "application"
    COMPONENT = "component"
    SURFACE_CODE = "surface_code"
    ERROR_CORRECTION = "error_correction"

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    type: BenchmarkType
    num_qubits: int
    circuit_depth: int
    num_shots: int = 1000
    error_threshold: float = 0.01
    timeout_seconds: float = 3600
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    success_rate: float
    error_rate: float
    execution_time: float
    fidelity: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

class BenchmarkManager:
    """Manage and execute quantum benchmarks"""
    
    def __init__(self, 
                 metrics_collector: Optional[MetricsCollector] = None,
                 error_analyzer: Optional[ErrorAnalyzer] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.error_analyzer = error_analyzer or ErrorAnalyzer()
        self.benchmark_history: Dict[BenchmarkType, List[BenchmarkResult]] = defaultdict(list)
        self.active_benchmarks: Dict[str, BenchmarkConfig] = {}

    async def run_benchmark(self, 
                          config: BenchmarkConfig,
                          circuit_generator: Callable) -> BenchmarkResult:
        """
        Run a quantum benchmark
        
        Args:
            config: Benchmark configuration
            circuit_generator: Function to generate test circuits
            
        Returns:
            BenchmarkResult containing performance metrics
        """
        try:
            start_time = datetime.now().timestamp()
            
            # Generate and run test circuits
            results = await self._execute_benchmark_circuits(
                config, circuit_generator
            )
            
            execution_time = datetime.now().timestamp() - start_time
            
            # Calculate metrics
            success_rate = results.get("success_rate", 0.0)
            error_rate = 1.0 - success_rate
            fidelity = results.get("fidelity")
            
            # Record results
            result = BenchmarkResult(
                success_rate=success_rate,
                error_rate=error_rate,
                execution_time=execution_time,
                fidelity=fidelity,
                metadata=results.get("metadata")
            )
            
            self.benchmark_history[config.type].append(result)
            
            # Update metrics
            self._update_metrics(config, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            raise

    async def run_surface_code_benchmark(self, 
                                       distance: int,
                                       num_cycles: int) -> Dict[str, Any]:
        """
        Run surface code specific benchmarks based on Google's paper
        
        Args:
            distance: Code distance
            num_cycles: Number of QEC cycles
            
        Returns:
            Dictionary containing benchmark results
        """
        try:
            config = BenchmarkConfig(
                type=BenchmarkType.SURFACE_CODE,
                num_qubits=distance * distance,
                circuit_depth=num_cycles,
                num_shots=1000
            )
            
            # Run stabilizer measurements
            stabilizer_results = await self._measure_stabilizers(distance, num_cycles)
            
            # Analyze logical error rate
            logical_error_rate = self._analyze_logical_errors(stabilizer_results)
            
            # Record results
            result = BenchmarkResult(
                success_rate=1.0 - logical_error_rate,
                error_rate=logical_error_rate,
                execution_time=stabilizer_results["execution_time"],
                metadata={
                    "distance": distance,
                    "num_cycles": num_cycles,
                    "stabilizer_data": stabilizer_results
                }
            )
            
            self.benchmark_history[BenchmarkType.SURFACE_CODE].append(result)
            
            return {
                "logical_error_rate": logical_error_rate,
                "physical_error_rates": stabilizer_results["error_rates"],
                "execution_metrics": {
                    "time": result.execution_time,
                    "success_rate": result.success_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Error running surface code benchmark: {str(e)}")
            raise

    def analyze_benchmark_trends(self, 
                               benchmark_type: BenchmarkType,
                               window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze trends in benchmark results
        
        Args:
            benchmark_type: Type of benchmark to analyze
            window_size: Optional window for recent results
            
        Returns:
            Dictionary containing trend analysis
        """
        try:
            results = self.benchmark_history[benchmark_type]
            if window_size:
                results = results[-window_size:]
                
            if not results:
                return {}
                
            error_rates = [r.error_rate for r in results]
            times = [r.timestamp for r in results]
            
            # Calculate trends
            coeffs = np.polyfit(times, error_rates, 1)
            trend = {
                "slope": float(coeffs[0]),
                "intercept": float(coeffs[1]),
                "direction": "improving" if coeffs[0] < 0 else "degrading"
            }
            
            # Calculate statistics
            stats = {
                "mean_error_rate": float(np.mean(error_rates)),
                "std_error_rate": float(np.std(error_rates)),
                "mean_execution_time": float(np.mean([r.execution_time for r in results]))
            }
            
            return {
                "trend": trend,
                "stats": stats,
                "sample_size": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing benchmark trends: {str(e)}")
            return {}

    async def _execute_benchmark_circuits(self,
                                       config: BenchmarkConfig,
                                       circuit_generator: Callable) -> Dict[str, Any]:
        """Execute benchmark circuits and gather results"""
        try:
            circuits = circuit_generator(config)
            results = defaultdict(list)
            
            for circuit in circuits:
                # Execute circuit and collect metrics
                circuit_result = await self._run_circuit(circuit, config)
                results["success_rates"].append(circuit_result["success_rate"])
                results["fidelities"].append(circuit_result.get("fidelity", 0.0))
                
            return {
                "success_rate": float(np.mean(results["success_rates"])),
                "fidelity": float(np.mean(results["fidelities"])),
                "metadata": {
                    "circuit_count": len(circuits),
                    "individual_results": results
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing benchmark circuits: {str(e)}")
            raise

    async def _measure_stabilizers(self,
                                 distance: int,
                                 num_cycles: int) -> Dict[str, Any]:
        """
        Measure surface code stabilizers
        
        Args:
            distance: Code distance
            num_cycles: Number of QEC cycles
            
        Returns:
            Dictionary containing stabilizer measurement results
        """
        try:
            start_time = datetime.now().timestamp()
            
            # Initialize results storage
            stabilizer_results = {
                "X_stabilizers": [],
                "Z_stabilizers": [],
                "error_rates": [],
                "detection_events": []
            }
            
            # Calculate number of stabilizers
            num_stabilizers = distance * distance - 1
            
            # Run QEC cycles
            for cycle in range(num_cycles):
                cycle_results = await self._execute_stabilizer_cycle(
                    distance=distance,
                    cycle_index=cycle
                )
                
                # Record X and Z stabilizer measurements
                stabilizer_results["X_stabilizers"].append(
                    cycle_results["X_measurements"]
                )
                stabilizer_results["Z_stabilizers"].append(
                    cycle_results["Z_measurements"]
                )
                
                # Track detection events
                detection_events = self._analyze_detection_events(
                    previous=stabilizer_results["detection_events"][-1] if cycle > 0 else None,
                    current=cycle_results
                )
                stabilizer_results["detection_events"].append(detection_events)
                
                # Calculate error rate for this cycle
                cycle_error_rate = len([e for e in detection_events if e]) / num_stabilizers
                stabilizer_results["error_rates"].append(cycle_error_rate)
            
            return {
                "execution_time": datetime.now().timestamp() - start_time,
                "stabilizer_results": stabilizer_results,
                "error_rates": {
                    "mean": float(np.mean(stabilizer_results["error_rates"])),
                    "std": float(np.std(stabilizer_results["error_rates"])),
                    "per_cycle": stabilizer_results["error_rates"]
                },
                "detection_statistics": self._calculate_detection_statistics(
                    stabilizer_results["detection_events"]
                )
            }
            
        except Exception as e:
            logger.error(f"Error measuring stabilizers: {str(e)}")
            raise

    async def _execute_stabilizer_cycle(self,
                                      distance: int,
                                      cycle_index: int) -> Dict[str, Any]:
        """
        Execute a single QEC cycle of stabilizer measurements
        
        Args:
            distance: Code distance
            cycle_index: Current cycle number
            
        Returns:
            Dictionary containing cycle measurements
        """
        try:
            # Initialize measurement results
            x_measurements = np.zeros((distance-1, distance-1), dtype=int)
            z_measurements = np.zeros((distance-1, distance-1), dtype=int)
            
            # Perform X-type stabilizer measurements
            for i in range(distance-1):
                for j in range(distance-1):
                    x_measurements[i,j] = await self._measure_x_stabilizer(i, j, distance)
            
            # Perform Z-type stabilizer measurements
            for i in range(distance-1):
                for j in range(distance-1):
                    z_measurements[i,j] = await self._measure_z_stabilizer(i, j, distance)
                    
            return {
                "X_measurements": x_measurements.tolist(),
                "Z_measurements": z_measurements.tolist(),
                "cycle_index": cycle_index
            }
            
        except Exception as e:
            logger.error(f"Error executing stabilizer cycle: {str(e)}")
            raise

    def _analyze_logical_errors(self,
                              stabilizer_results: Dict[str, Any]) -> float:
        """
        Analyze logical error rate from stabilizer measurements
        
        Args:
            stabilizer_results: Results from stabilizer measurements
            
        Returns:
            Calculated logical error rate
        """
        try:
            detection_events = stabilizer_results["stabilizer_results"]["detection_events"]
            error_chains = self._identify_error_chains(detection_events)
            
            # Count logical errors
            logical_x_errors = self._count_logical_x_errors(error_chains)
            logical_z_errors = self._count_logical_z_errors(error_chains)
            
            total_cycles = len(detection_events)
            logical_error_rate = (logical_x_errors + logical_z_errors) / (2 * total_cycles)
            
            return float(logical_error_rate)
            
        except Exception as e:
            logger.error(f"Error analyzing logical errors: {str(e)}")
            raise

    def _identify_error_chains(self,
                             detection_events: List[List[bool]]) -> List[List[Tuple[int, int]]]:
        """
        Identify chains of errors from detection events
        
        Args:
            detection_events: List of detection events per cycle
            
        Returns:
            List of error chains (sequences of coordinates)
        """
        try:
            error_chains = []
            visited = set()
            
            for cycle_idx, cycle_events in enumerate(detection_events):
                for event_idx, is_error in enumerate(cycle_events):
                    if is_error and (cycle_idx, event_idx) not in visited:
                        chain = self._trace_error_chain(
                            detection_events, cycle_idx, event_idx, visited
                        )
                        if chain:
                            error_chains.append(chain)
            
            return error_chains
            
        except Exception as e:
            logger.error(f"Error identifying error chains: {str(e)}")
            return []

    def _trace_error_chain(self,
                          detection_events: List[List[bool]],
                          start_cycle: int,
                          start_idx: int,
                          visited: set) -> List[Tuple[int, int]]:
        """
        Trace a chain of errors starting from a detection event
        
        Args:
            detection_events: All detection events
            start_cycle: Starting cycle index
            start_idx: Starting event index
            visited: Set of visited coordinates
            
        Returns:
            List of coordinates forming the error chain
        """
        try:
            chain = [(start_cycle, start_idx)]
            visited.add((start_cycle, start_idx))
            
            # Look for connected errors in space and time
            neighbors = self._get_neighboring_events(
                start_cycle, start_idx, len(detection_events)
            )
            
            for next_cycle, next_idx in neighbors:
                if (next_cycle, next_idx) not in visited and \
                   detection_events[next_cycle][next_idx]:
                    sub_chain = self._trace_error_chain(
                        detection_events, next_cycle, next_idx, visited
                    )
                    chain.extend(sub_chain)
            
            return chain
            
        except Exception as e:
            logger.error(f"Error tracing error chain: {str(e)}")
            return []

    def _get_neighboring_events(self,
                              cycle: int,
                              idx: int,
                              num_cycles: int) -> List[Tuple[int, int]]:
        """Get neighboring event coordinates in space and time"""
        neighbors = []
        
        # Same cycle neighbors
        for delta_idx in [-1, 1]:
            new_idx = idx + delta_idx
            if new_idx >= 0:  # Add appropriate upper bound check based on layout
                neighbors.append((cycle, new_idx))
        
        # Adjacent cycle neighbors
        for delta_cycle in [-1, 1]:
            new_cycle = cycle + delta_cycle
            if 0 <= new_cycle < num_cycles:
                neighbors.append((new_cycle, idx))
        
        return neighbors

    def _calculate_detection_statistics(self,
                                     detection_events: List[List[bool]]) -> Dict[str, float]:
        """
        Calculate statistics about detection events
        
        Args:
            detection_events: List of detection events per cycle
            
        Returns:
            Dictionary containing detection statistics
        """
        try:
            total_events = sum(sum(cycle) for cycle in detection_events)
            total_measurements = sum(len(cycle) for cycle in detection_events)
            
            consecutive_events = 0
            for cycle in detection_events:
                for i in range(len(cycle)-1):
                    if cycle[i] and cycle[i+1]:
                        consecutive_events += 1
            
            return {
                "event_rate": float(total_events / total_measurements),
                "consecutive_event_rate": float(consecutive_events / total_events) if total_events > 0 else 0.0,
                "total_events": total_events,
                "total_measurements": total_measurements
            }
            
        except Exception as e:
            logger.error(f"Error calculating detection statistics: {str(e)}")
            return {}

    async def _measure_x_stabilizer(self,
                                  i: int,
                                  j: int,
                                  distance: int) -> int:
        """
        Measure X-type stabilizer at coordinates (i,j)
        
        Args:
            i, j: Coordinates in the surface code lattice
            distance: Code distance
            
        Returns:
            Measurement result (0 or 1)
        """
        try:
            # Get involved data qubits
            data_qubits = self._get_x_stabilizer_qubits(i, j, distance)
            
            # Prepare ancilla in |+⟩ state
            ancilla_index = self._get_ancilla_index(i, j, distance)
            
            # Apply CNOT gates between ancilla and data qubits
            for qubit in data_qubits:
                # CNOT with ancilla as control
                await self._apply_cnot(ancilla_index, qubit)
            
            # Measure ancilla in X-basis
            result = await self._measure_in_x_basis(ancilla_index)
            return result
            
        except Exception as e:
            logger.error(f"Error measuring X stabilizer: {str(e)}")
            raise

    async def _measure_z_stabilizer(self,
                                  i: int,
                                  j: int,
                                  distance: int) -> int:
        """
        Measure Z-type stabilizer at coordinates (i,j)
        
        Args:
            i, j: Coordinates in the surface code lattice
            distance: Code distance
            
        Returns:
            Measurement result (0 or 1)
        """
        try:
            # Get involved data qubits
            data_qubits = self._get_z_stabilizer_qubits(i, j, distance)
            
            # Prepare ancilla in |0⟩ state
            ancilla_index = self._get_ancilla_index(i, j, distance)
            
            # Apply CNOT gates between data qubits and ancilla
            for qubit in data_qubits:
                # CNOT with data qubit as control
                await self._apply_cnot(qubit, ancilla_index)
            
            # Measure ancilla in Z-basis
            result = await self._measure_in_z_basis(ancilla_index)
            return result
            
        except Exception as e:
            logger.error(f"Error measuring Z stabilizer: {str(e)}")
            raise

    def _is_logical_x_error(self, chain: List[Tuple[int, int]]) -> bool:
        """
        Determine if an error chain constitutes a logical X error
        
        Args:
            chain: List of error coordinates
            
        Returns:
            True if chain corresponds to logical X error
        """
        try:
            # Get endpoints of the chain
            if not chain:
                return False
                
            start = chain[0]
            end = chain[-1]
            
            # Check if chain spans the lattice in the Z-logical direction
            spans_logical_z = self._spans_logical_z_direction(start, end)
            
            # Check if chain has odd parity
            odd_parity = len(chain) % 2 == 1
            
            return spans_logical_z and odd_parity
            
        except Exception as e:
            logger.error(f"Error checking logical X error: {str(e)}")
            return False

    def _is_logical_z_error(self, chain: List[Tuple[int, int]]) -> bool:
        """
        Determine if an error chain constitutes a logical Z error
        
        Args:
            chain: List of error coordinates
            
        Returns:
            True if chain corresponds to logical Z error
        """
        try:
            # Get endpoints of the chain
            if not chain:
                return False
                
            start = chain[0]
            end = chain[-1]
            
            # Check if chain spans the lattice in the X-logical direction
            spans_logical_x = self._spans_logical_x_direction(start, end)
            
            # Check if chain has odd parity
            odd_parity = len(chain) % 2 == 1
            
            return spans_logical_x and odd_parity
            
        except Exception as e:
            logger.error(f"Error checking logical Z error: {str(e)}")
            return False

    def _get_x_stabilizer_qubits(self,
                                i: int,
                                j: int,
                                distance: int) -> List[int]:
        """Get data qubits involved in X-type stabilizer"""
        try:
            # Calculate physical qubit indices for X stabilizer
            # Based on surface code layout from Google paper
            base_index = i * distance + j
            return [
                base_index,
                base_index + 1,
                base_index + distance,
                base_index + distance + 1
            ]
        except Exception as e:
            logger.error(f"Error getting X stabilizer qubits: {str(e)}")
            return []

    def _get_z_stabilizer_qubits(self,
                                i: int,
                                j: int,
                                distance: int) -> List[int]:
        """Get data qubits involved in Z-type stabilizer"""
        try:
            # Calculate physical qubit indices for Z stabilizer
            # Based on surface code layout
            base_index = i * distance + j
            return [
                base_index,
                base_index + 1,
                base_index + distance,
                base_index + distance + 1
            ]
        except Exception as e:
            logger.error(f"Error getting Z stabilizer qubits: {str(e)}")
            return []

    def _get_ancilla_index(self,
                          i: int,
                          j: int,
                          distance: int) -> int:
        """Calculate ancilla qubit index for stabilizer measurement"""
        try:
            # Calculate ancilla index based on surface code layout
            return distance * distance + i * (distance - 1) + j
        except Exception as e:
            logger.error(f"Error calculating ancilla index: {str(e)}")
            raise

    def _spans_logical_x_direction(self,
                                 start: Tuple[int, int],
                                 end: Tuple[int, int]) -> bool:
        """Check if error chain spans logical X direction"""
        try:
            start_x = start[1]  # x coordinate
            end_x = end[1]
            
            # Check if chain connects opposite boundaries
            return (start_x == 0 and end_x == self.code_distance - 1) or \
                   (start_x == self.code_distance - 1 and end_x == 0)
        except Exception as e:
            logger.error(f"Error checking X direction span: {str(e)}")
            return False

    def _spans_logical_z_direction(self,
                                 start: Tuple[int, int],
                                 end: Tuple[int, int]) -> bool:
        """Check if error chain spans logical Z direction"""
        try:
            start_z = start[0]  # z coordinate
            end_z = end[0]
            
            # Check if chain connects opposite boundaries
            return (start_z == 0 and end_z == self.code_distance - 1) or \
                   (start_z == self.code_distance - 1 and end_z == 0)
        except Exception as e:
            logger.error(f"Error checking Z direction span: {str(e)}")
            return False

    async def _apply_cnot(self,
                         control: int,
                         target: int) -> None:
        """Apply CNOT gate between control and target qubits"""
        try:
            # Implementation will depend on your quantum hardware interface
            # This is a placeholder for the actual CNOT implementation
            pass
        except Exception as e:
            logger.error(f"Error applying CNOT gate: {str(e)}")
            raise

    async def _measure_in_x_basis(self, qubit: int) -> int:
        """Measure qubit in X basis"""
        try:
            # Implementation will depend on your quantum hardware interface
            # This is a placeholder for actual X-basis measurement
            return np.random.randint(2)
        except Exception as e:
            logger.error(f"Error measuring in X basis: {str(e)}")
            raise

    async def _measure_in_z_basis(self, qubit: int) -> int:
        """Measure qubit in Z basis"""
        try:
            # Implementation will depend on your quantum hardware interface
            # This is a placeholder for actual Z-basis measurement
            return np.random.randint(2)
        except Exception as e:
            logger.error(f"Error measuring in Z basis: {str(e)}")
            raise