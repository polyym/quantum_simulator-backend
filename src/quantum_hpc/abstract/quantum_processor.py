# src/quantum_hpc/abstract/quantum_processor.py

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np

# Import our utility modules
from src.utils.error_analysis import ErrorAnalyzer, ErrorType, ErrorMetrics
from src.utils.metrics_collection import MetricsCollector, MetricType, MetricValue
from src.utils.benchmarking import BenchmarkManager

logger = logging.getLogger(__name__)

class ProcessorType(Enum):
    """Types of quantum processors supported."""
    PHYSICAL = "physical"          # Real quantum hardware
    VIRTUAL = "virtual"            # Simulated quantum processor
    DISTRIBUTED = "distributed"    # Distributed quantum system
    SURFACE_CODE = "surface_code"  # Surface code logical processor
    HYBRID = "hybrid"              # Hybrid quantum-classical processor

class GateType(Enum):
    """Supported quantum gate types."""
    SINGLE_QUBIT = "single_qubit"          # Single-qubit operations
    TWO_QUBIT = "two_qubit"                # Two-qubit operations
    MULTI_QUBIT = "multi_qubit"            # Multi-qubit operations
    MEASUREMENT = "measurement"            # Measurement operations
    LOGICAL = "logical"                    # Logical qubit operations
    ERROR_CORRECTION = "error_correction"  # Error correction operations

class ErrorModel(Enum):
    """Supported error models."""
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    THERMAL = "thermal"
    CROSSTALK = "crosstalk"
    CUSTOM = "custom"

@dataclass
class ProcessorCapabilities:
    """Quantum processor capabilities and constraints."""
    num_qubits: int
    connectivity_map: Dict[int, List[int]]  # Qubit connectivity
    gate_set: Dict[str, GateType]           # Available gates
    max_circuit_depth: Optional[int] = None
    coherence_time: Optional[float] = None
    gate_times: Dict[str, float] = field(default_factory=dict)   # Gate execution times
    native_gates: List[str] = field(default_factory=list)        # Hardware-native gates
    error_rates: Dict[str, float] = field(default_factory=dict)  # Gate error rates
    measurement_fidelity: float = 0.99
    reset_fidelity: float = 0.99
    supports_feedback: bool = False
    max_parallel_ops: Optional[int] = None

@dataclass
class ProcessorMetrics:
    """Runtime metrics for processor performance."""
    gate_fidelities: Dict[str, float]
    readout_fidelities: Dict[int, float]
    error_rates: Dict[str, float]
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    logical_error_rate: Optional[float] = None
    physical_error_rates: Optional[Dict[str, float]] = None
    coherence_metrics: Optional[Dict[str, float]] = None
    successful_operations: int = 0
    failed_operations: int = 0

@dataclass
class QuantumState:
    """Quantum state representation."""
    state_vector: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    num_qubits: int = 0
    is_mixed: bool = False
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

class QuantumProcessor(ABC):
    """
    Abstract base class for quantum processors.
    Defines interface for different types of quantum processors.
    """
    
    def __init__(self, 
                 processor_type: ProcessorType,
                 capabilities: ProcessorCapabilities):
        """
        Initialize quantum processor with given capabilities.
        
        Args:
            processor_type: Type of quantum processor
            capabilities: Processor capabilities and constraints
        """
        self.processor_type = processor_type
        self.capabilities = capabilities
        
        # Initialize metrics and analysis tools
        self.metrics_collector = MetricsCollector()
        self.error_analyzer = ErrorAnalyzer()
        self.benchmark_manager = BenchmarkManager(
            metrics_collector=self.metrics_collector,
            error_analyzer=self.error_analyzer
        )
        
        # State management
        self._state = QuantumState(num_qubits=capabilities.num_qubits)
        self._active = False
        self._error_model = None
        self._calibration_data = {}
        
        # Operation history
        self._operation_history: List[Dict[str, Any]] = []
        self._error_history: List[Dict[str, Any]] = []
        
        # Initialize metric thresholds
        self._init_metric_thresholds()

    def _init_metric_thresholds(self):
        """Initialize metric monitoring thresholds."""
        # Example threshold for extremely low logical error rates
        self.metrics_collector.set_threshold(
            MetricType.LOGICAL_ERROR_RATE,
            threshold=1e-6,  # Target error rate for long-lived qubit
            callback=self._handle_error_threshold_exceeded
        )
        
        # Additional threshold for gate fidelity
        self.metrics_collector.set_threshold(
            MetricType.GATE_FIDELITY,
            threshold=0.999,
            callback=self._handle_low_fidelity
        )

    @abstractmethod
    async def initialize(self, 
                         initial_state: Optional[Union[np.ndarray, str]] = None) -> bool:
        """
        Initialize quantum processor to given state.
        
        Args:
            initial_state: Optional initial state specification
                          (state vector, density matrix, or descriptive string)
            
        Returns:
            bool: Success status
        """
        pass

    @abstractmethod
    async def apply_gate(self, 
                         gate: str,
                         qubits: List[int],
                         params: Optional[Dict] = None) -> bool:
        """
        Apply quantum gate operation.
        
        Args:
            gate: Name/identifier of the gate
            qubits: Target qubits
            params: Optional gate parameters
            
        Returns:
            bool: Success status
        """
        if not self._active:
            raise ProcessorNotActiveError("Processor not active")
            
        if not await self.validate_operation(gate, qubits):
            raise InvalidOperationError(f"Invalid operation: {gate} on qubits {qubits}")
            
        try:
            # Record the operation
            self._operation_history.append({
                'gate': gate,
                'qubits': qubits,
                'params': params,
                'timestamp': datetime.now().timestamp()
            })
            
            # Apply error model if present
            if self._error_model:
                self._apply_error_model(gate, qubits)
                
            return True
            
        except Exception as e:
            logger.error(f"Error applying gate '{gate}' on qubits {qubits}: {str(e)}")
            self._handle_operation_error(gate, qubits, str(e))
            return False

    @abstractmethod
    async def measure(self,
                      qubits: List[int],
                      basis: Optional[str] = "Z") -> Tuple[List[int], float]:
        """
        Measure specified qubits in a given basis.
        
        Args:
            qubits: Qubits to measure
            basis: Measurement basis ("Z", "X", or "Y")
            
        Returns:
            Tuple of (measurement results, fidelity)
        """
        if not self._active:
            raise ProcessorNotActiveError("Processor not active")
            
        try:
            # Apply measurement error model if set
            if self._error_model:
                self._apply_measurement_errors(qubits)
                
            self._operation_history.append({
                'operation': 'measure',
                'qubits': qubits,
                'basis': basis,
                'timestamp': datetime.now().timestamp()
            })
            
            # Concrete classes will return actual measurement outcomes and fidelity
            return [], 0.0
            
        except Exception as e:
            logger.error(f"Error measuring qubits {qubits} in {basis} basis: {str(e)}")
            self._handle_operation_error('measure', qubits, str(e))
            raise

    @abstractmethod
    async def reset(self,
                    qubits: Optional[List[int]] = None) -> bool:
        """
        Reset specified qubits or the entire processor.
        
        Args:
            qubits: Optional list of qubits to reset. If None, resets all.
            
        Returns:
            bool: Success status
        """
        try:
            if qubits is None:
                qubits = list(range(self.capabilities.num_qubits))
                
            # Apply reset error model if set
            if self._error_model:
                self._apply_reset_errors(qubits)
                
            self._operation_history.append({
                'operation': 'reset',
                'qubits': qubits,
                'timestamp': datetime.now().timestamp()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting qubits {qubits}: {str(e)}")
            self._handle_operation_error('reset', qubits, str(e))
            return False

    async def validate_operation(self,
                                 gate: str,
                                 qubits: List[int]) -> bool:
        """
        Validate if an operation is supported by this processor.
        
        Args:
            gate: Gate operation name/identifier
            qubits: Target qubits
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if gate is supported
            if gate not in self.capabilities.gate_set:
                logger.error(f"Gate '{gate}' not in supported gate set")
                return False
                
            # Check qubit indices
            if not all(q < self.capabilities.num_qubits for q in qubits):
                logger.error("Invalid qubit indices in operation")
                return False
                
            # For multi-qubit gates, verify connectivity
            if len(qubits) > 1:
                if not self._check_connectivity(qubits):
                    logger.error("Qubits not connected for multi-qubit gate")
                    return False
                    
            # Check circuit depth constraints
            if (self.capabilities.max_circuit_depth is not None and 
                len(self._operation_history) >= self.capabilities.max_circuit_depth):
                logger.error("Maximum circuit depth exceeded")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating operation '{gate}' on qubits {qubits}: {str(e)}")
            return False

    def _check_connectivity(self, qubits: List[int]) -> bool:
        """Check if qubits are connected according to the processor's connectivity map."""
        for i in range(len(qubits) - 1):
            q1, q2 = qubits[i], qubits[i + 1]
            if q2 not in self.capabilities.connectivity_map[q1]:
                return False
        return True

    def get_metrics(self) -> Optional[ProcessorMetrics]:
        """
        Compile the latest metrics into a ProcessorMetrics object.
        
        Returns:
            ProcessorMetrics containing gate fidelities, error rates, etc.
            or None if no metrics have been recorded yet.
        """
        if not self.metrics_collector.metrics:
            return None
            
        gate_fidelities = {}
        readout_fidelities = {}
        error_rates = {}
        
        # Iterate over each metric type and pick the latest measurement
        for metric_type, values in self.metrics_collector.metrics.items():
            if not values:
                continue
                
            latest = values[-1]
            if metric_type == MetricType.GATE_FIDELITY and latest.metadata:
                gate_name = latest.metadata.get('gate')
                if gate_name:
                    gate_fidelities[gate_name] = latest.value
            elif metric_type == MetricType.READOUT_FIDELITY and latest.metadata:
                qubit_id = latest.metadata.get('qubit')
                if qubit_id is not None:
                    readout_fidelities[qubit_id] = latest.value
            elif metric_type == MetricType.PHYSICAL_ERROR_RATE and latest.metadata:
                err_type = latest.metadata.get('type')
                if err_type:
                    error_rates[err_type] = latest.value
        
        return ProcessorMetrics(
            gate_fidelities=gate_fidelities,
            readout_fidelities=readout_fidelities,
            error_rates=error_rates,
            logical_error_rate=self._calculate_logical_error_rate(),
            physical_error_rates=self._get_physical_error_rates(),
            coherence_metrics=self._get_coherence_metrics()
        )

    def _calculate_logical_error_rate(self) -> float:
        """
        Approximate logical error rate based on recent error events
        and operation history.
        """
        try:
            if not self._error_history:
                return 0.0
            recent_errors = self._error_history[-1000:]  # Window
            total_ops = len(self._operation_history[-1000:])
            return len(recent_errors) / total_ops if total_ops > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating logical error rate: {str(e)}")
            return 0.0

    def _get_physical_error_rates(self) -> Dict[str, float]:
        """Return current physical error rates from the capabilities."""
        try:
            return {
                'single_qubit': self.capabilities.error_rates.get('single_qubit', 0.0),
                'two_qubit': self.capabilities.error_rates.get('two_qubit', 0.0),
                'measurement': self.capabilities.error_rates.get('measurement', 0.0),
                'idle': self.capabilities.error_rates.get('idle', 0.0)
            }
        except Exception as e:
            logger.error(f"Error getting physical error rates: {str(e)}")
            return {}

    def _get_coherence_metrics(self) -> Dict[str, float]:
        """
        Return approximate coherence metrics (T1, T2, etc.)
        based on the processor's capabilities.
        """
        try:
            t_coh = self.capabilities.coherence_time or 0.0
            gate_times = self.capabilities.gate_times.values() or []
            avg_gate_time = np.mean(list(gate_times)) if gate_times else 0.0
            return {
                'T1': t_coh,
                'T2': t_coh * 0.7,  # Example ratio
                'gate_time': avg_gate_time
            }
        except Exception as e:
            logger.error(f"Error getting coherence metrics: {str(e)}")
            return {}

    def _handle_operation_error(self, operation: str, qubits: List[int], error_msg: str):
        """Record and log an error event related to a specific operation."""
        error_data = {
            'operation': operation,
            'qubits': qubits,
            'error': error_msg,
            'timestamp': datetime.now().timestamp()
        }
        self._error_history.append(error_data)
        # Mark this as a gate or measurement error
        err_type = ErrorType.GATE if operation != 'measure' else ErrorType.MEASUREMENT
        self.error_analyzer.record_error(
            error_rate=1.0,  # This operation failed
            error_type=err_type,
            metadata=error_data
        )

    def _handle_error_threshold_exceeded(self, 
                                         metric_type: MetricType,
                                         value: float,
                                         threshold: float):
        """Callback when a monitored metric (like LOGICAL_ERROR_RATE) exceeds its threshold."""
        logger.warning(
            f"Error threshold exceeded: {metric_type.value} = {value} (threshold: {threshold})"
        )
        self.error_analyzer.record_error(
            error_rate=value,
            error_type=ErrorType.LOGICAL if metric_type == MetricType.LOGICAL_ERROR_RATE else ErrorType.PHYSICAL,
            metadata={'threshold': threshold, 'metric_type': metric_type.value}
        )
        # Optionally trigger automatic calibration
        if self.capabilities.supports_feedback:
            asyncio.create_task(self.calibrate())

    def _handle_low_fidelity(self,
                             metric_type: MetricType,
                             value: float,
                             threshold: float):
        """
        Callback when gate fidelity drops below threshold.
        Collect further diagnostics and record the event.
        """
        logger.warning(
            f"Low fidelity detected: {metric_type.value} = {value} (threshold: {threshold})"
        )
        recent_ops = self._operation_history[-100:]
        problematic_gates = self._analyze_problematic_operations(recent_ops)
        
        # Record additional metrics
        self.metrics_collector.record_metric(
            metric_type=MetricType.GATE_FIDELITY,
            value=value,
            metadata={
                'problematic_gates': problematic_gates,
                'recent_operations': len(recent_ops)
            }
        )

    def _analyze_problematic_operations(self, operations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Scan recent operations for patterns that might correlate with errors.
        
        Returns:
            A dictionary mapping gate -> error rate.
        """
        gate_errors = defaultdict(int)
        gate_counts = defaultdict(int)
        for op in operations:
            gate_name = op.get('gate')
            if gate_name:
                gate_counts[gate_name] += 1
                # Check if there was a matching error event
                error_time = next(
                    (err['timestamp'] for err in self._error_history 
                     if abs(err['timestamp'] - op['timestamp']) < 1e-6),
                    None
                )
                if error_time is not None:
                    gate_errors[gate_name] += 1
        
        # Compute error rates per gate
        return {
            g: (gate_errors[g] / gate_counts[g])
            for g in gate_counts
            if gate_counts[g] > 0
        }

    @abstractmethod
    async def calibrate(self) -> Dict[str, float]:
        """
        Perform processor calibration.
        
        Returns:
            A dictionary of calibration results or parameters.
        """
        try:
            start_time = datetime.now().timestamp()
            # Simple calibration: reset all qubits, measure baseline metrics, etc.
            await self.reset()
            baseline_metrics = self.get_metrics()
            self._calibration_data = {
                'timestamp': start_time,
                'duration': datetime.now().timestamp() - start_time,
                'baseline_metrics': baseline_metrics,
            }
            return self._calibration_data
        except Exception as e:
            logger.error(f"Error during calibration: {str(e)}")
            raise

    @abstractmethod
    async def set_error_model(self, 
                              model_type: ErrorModel,
                              params: Dict[str, Any]) -> bool:
        """
        Set the current error model for the processor.
        
        Args:
            model_type: Which error model to apply
            params: Model-specific parameters
            
        Returns:
            bool indicating success or failure
        """
        try:
            self._error_model = {
                'type': model_type,
                'params': params,
                'timestamp': datetime.now().timestamp()
            }
            # Update capabilities with relevant error rates
            self.capabilities.error_rates.update(self._calculate_error_rates_from_model())
            return True
        except Exception as e:
            logger.error(f"Error setting error model {model_type}: {str(e)}")
            return False

    def _calculate_error_rates_from_model(self) -> Dict[str, float]:
        """
        Derive approximate error rates from the active error model's parameters.
        This can be specialized or extended based on the model type.
        """
        try:
            if not self._error_model:
                return {}
            model_type = self._error_model['type']
            params = self._error_model['params']
            
            if model_type == ErrorModel.DEPOLARIZING:
                return {
                    'single_qubit': params.get('p1', 0.0),
                    'two_qubit': params.get('p2', 0.0),
                    'measurement': params.get('p_meas', 0.0)
                }
            elif model_type == ErrorModel.AMPLITUDE_DAMPING:
                idle_time = self.capabilities.gate_times.get('idle', 0.0)
                t1 = params.get('T1', float('inf'))
                return {
                    'single_qubit': params.get('gamma', 0.0),
                    'idle': 1 - np.exp(-idle_time / t1) if t1 > 0 else 1.0
                }
            # Additional model handling (phase damping, thermal, etc.) can go here.
            return {}
        except Exception as e:
            logger.error(f"Error calculating error rates from model: {str(e)}")
            return {}

    @abstractmethod
    async def get_state(self) -> QuantumState:
        """
        Retrieve the current quantum state.
        Implementation will differ for physical vs. simulated processors.
        
        Returns:
            QuantumState object representing the processor's state.
        """
        if not self._active:
            raise ProcessorNotActiveError("Processor not active")
        return self._state

    async def get_operation_history(self, 
                                    start_time: Optional[float] = None,
                                    end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve the operation history in a given time window.
        
        Args:
            start_time: Filter operations after this UNIX timestamp
            end_time: Filter operations before this UNIX timestamp
        
        Returns:
            A list of recorded operations.
        """
        try:
            history = self._operation_history
            if start_time is not None:
                history = [op for op in history if op['timestamp'] >= start_time]
            if end_time is not None:
                history = [op for op in history if op['timestamp'] <= end_time]
            return history
        except Exception as e:
            logger.error(f"Error retrieving operation history: {str(e)}")
            return []

    async def get_error_history(self,
                                error_type: Optional[ErrorType] = None) -> List[Dict[str, Any]]:
        """
        Retrieve error events, optionally filtered by a specific error type.
        
        Args:
            error_type: Filter error events of this type if provided
        
        Returns:
            A list of error event records.
        """
        try:
            history = self._error_history
            if error_type is not None:
                history = [err for err in history if err.get('error_type') == error_type]
            return history
        except Exception as e:
            logger.error(f"Error retrieving error history: {str(e)}")
            return []

    # Placeholder methods for applying error models within gate/measurement/reset
    # Child classes can override these for more advanced or realistic simulation.
    def _apply_error_model(self, gate: str, qubits: List[int]) -> None:
        """Apply gate-based error model logic if needed."""
        pass

    def _apply_measurement_errors(self, qubits: List[int]) -> None:
        """Apply measurement-based error model logic if needed."""
        pass

    def _apply_reset_errors(self, qubits: List[int]) -> None:
        """Apply reset-based error model logic if needed."""
        pass

# Helper class to implement various error channels
class ErrorModelImplementation:
    """Implementation of different error models (depolarizing, amplitude damping, etc.)."""

    @staticmethod
    def apply_depolarizing_error(state: QuantumState, 
                                 prob: float,
                                 qubits: List[int]) -> QuantumState:
        """Apply depolarizing noise to a quantum state."""
        try:
            if state.is_mixed and state.density_matrix is not None:
                # Density matrix channel
                rho = state.density_matrix
                for _ in qubits:
                    rho = (1 - prob) * rho + \
                          (prob / 3) * sum(pauli @ rho @ pauli.conj().T 
                                           for pauli in [X_GATE, Y_GATE, Z_GATE])
                state.density_matrix = rho
            elif state.state_vector is not None:
                # State vector approach
                for qubit in qubits:
                    if np.random.random() < prob:
                        error_type = np.random.choice(['X', 'Y', 'Z'])
                        gate_mat = X_GATE if error_type == 'X' else \
                                   Y_GATE if error_type == 'Y' else \
                                   Z_GATE
                        state.state_vector = apply_gate_to_qubit(gate_mat, state.state_vector, qubit)
            return state
        except Exception as e:
            logger.error(f"Error applying depolarizing noise: {str(e)}")
            return state

    @staticmethod
    def apply_amplitude_damping(state: QuantumState,
                                gamma: float,
                                qubits: List[int]) -> QuantumState:
        """Apply amplitude damping to a quantum state."""
        try:
            if state.is_mixed and state.density_matrix is not None:
                rho = state.density_matrix
                E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
                E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
                for _ in qubits:
                    rho = sum(Ei @ rho @ Ei.conj().T for Ei in [E0, E1])
                state.density_matrix = rho
            elif state.state_vector is not None:
                for qubit in qubits:
                    if np.random.random() < gamma:
                        state.state_vector = collapse_to_ground(state.state_vector, qubit)
            return state
        except Exception as e:
            logger.error(f"Error applying amplitude damping: {str(e)}")
            return state

    @staticmethod
    def apply_phase_damping(state: QuantumState,
                            gamma: float,
                            qubits: List[int]) -> QuantumState:
        """Apply phase damping to a quantum state."""
        try:
            if state.is_mixed and state.density_matrix is not None:
                rho = state.density_matrix
                E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
                E1 = np.array([[0, 0], [0, np.sqrt(gamma)]])
                for _ in qubits:
                    rho = sum(Ei @ rho @ Ei.conj().T for Ei in [E0, E1])
                state.density_matrix = rho
            elif state.state_vector is not None:
                for qubit in qubits:
                    if np.random.random() < gamma:
                        state.state_vector = apply_gate_to_qubit(Z_GATE, state.state_vector, qubit)
            return state
        except Exception as e:
            logger.error(f"Error applying phase damping: {str(e)}")
            return state

    @staticmethod
    def apply_thermal_noise(state: QuantumState,
                            temperature: float,
                            coupling: float,
                            qubits: List[int]) -> QuantumState:
        """
        Apply thermal noise based on temperature and coupling strength.
        Example approach to approximate hardware-specific noise.
        """
        try:
            n_thermal = 1 / (np.exp(1 / (temperature * coupling)) - 1)
            gamma_up = coupling * n_thermal
            gamma_down = coupling * (n_thermal + 1)

            if state.is_mixed and state.density_matrix is not None:
                rho = state.density_matrix
                L_up = np.array([[0, 1], [0, 0]])
                L_down = np.array([[0, 0], [1, 0]])
                for _ in qubits:
                    drho = (gamma_up * (L_up @ rho @ L_up.conj().T - 0.5 * (L_up.conj().T @ L_up @ rho + rho @ L_up.conj().T @ L_up)) +
                            gamma_down * (L_down @ rho @ L_down.conj().T - 0.5 * (L_down.conj().T @ L_down @ rho + rho @ L_down.conj().T @ L_down)))
                    rho += drho
                state.density_matrix = rho
            elif state.state_vector is not None:
                for qubit in qubits:
                    if np.random.random() < gamma_up:
                        state.state_vector = excite_qubit(state.state_vector, qubit)
                    if np.random.random() < gamma_down:
                        state.state_vector = collapse_to_ground(state.state_vector, qubit)
            return state
        except Exception as e:
            logger.error(f"Error applying thermal noise: {str(e)}")
            return state

    @staticmethod
    def apply_crosstalk(state: QuantumState,
                        coupling_map: Dict[Tuple[int, int], float],
                        active_qubits: List[int]) -> QuantumState:
        """Apply crosstalk effects (ZZ interactions) to active qubits."""
        try:
            if state.is_mixed and state.density_matrix is not None:
                rho = state.density_matrix
                for (q1, q2), strength in coupling_map.items():
                    if q1 in active_qubits or q2 in active_qubits:
                        H_int = strength * np.kron(Z_GATE, Z_GATE)
                        U = np.exp(-1j * H_int)
                        rho = U @ rho @ U.conj().T
                state.density_matrix = rho
            elif state.state_vector is not None:
                for (q1, q2), strength in coupling_map.items():
                    if q1 in active_qubits or q2 in active_qubits:
                        phase = np.exp(-1j * strength)
                        state.state_vector = apply_two_qubit_phase(state.state_vector, q1, q2, phase)
            return state
        except Exception as e:
            logger.error(f"Error applying crosstalk: {str(e)}")
            return state

# Helper functions for manipulating state vectors/density matrices
def apply_gate_to_qubit(gate: np.ndarray, 
                        state: np.ndarray,
                        qubit: int) -> np.ndarray:
    """Apply a single-qubit gate to a specific qubit index in a state vector."""
    n_qubits = int(np.log2(len(state)))
    op = np.eye(1)
    for i in range(n_qubits):
        op = np.kron(op, gate if i == qubit else np.eye(2))
    return op @ state

def apply_two_qubit_phase(state: np.ndarray,
                          q1: int,
                          q2: int,
                          phase: complex) -> np.ndarray:
    """Apply a phase shift to the |11> component of two qubits (q1, q2)."""
    new_state = state.copy()
    mask = (1 << q1) | (1 << q2)
    for i in range(len(state)):
        if (i & mask) == mask:
            new_state[i] *= phase
    return new_state

def collapse_to_ground(state: np.ndarray, qubit: int) -> np.ndarray:
    """Collapse a qubit to the |0> state."""
    new_state = np.zeros_like(state)
    for i in range(len(state)):
        if not (i & (1 << qubit)):
            new_state[i] = state[i]
    norm = np.sqrt(np.sum(np.abs(new_state)**2))
    if norm > 0:
        new_state /= norm
    return new_state

def excite_qubit(state: np.ndarray, qubit: int) -> np.ndarray:
    """Excite a qubit to the |1> state."""
    new_state = np.zeros_like(state)
    mask = 1 << qubit
    for i in range(len(state)):
        if (i & mask) == mask:
            new_state[i] = state[i]
    norm = np.sqrt(np.sum(np.abs(new_state)**2))
    if norm > 0:
        new_state /= norm
    return new_state

# Common single-qubit gates
X_GATE = np.array([[0, 1],
                   [1, 0]])
Y_GATE = np.array([[0, -1j],
                   [1j, 0]])
Z_GATE = np.array([[1, 0],
                   [0, -1]])

# Export symbols
__all__ = [
    'ProcessorType',
    'GateType',
    'ErrorModel',
    'ProcessorCapabilities',
    'ProcessorMetrics',
    'QuantumState',
    'QuantumProcessor',
    'ProcessorError',
    'InvalidOperationError',
    'ProcessorNotActiveError',
    'ErrorModelImplementation'
]

# Custom exceptions
class ProcessorError(Exception):
    """General exception for processor-related errors."""
    pass

class InvalidOperationError(ProcessorError):
    """Raised when an invalid operation is requested on the processor."""
    pass

class ProcessorNotActiveError(ProcessorError):
    """Raised when an operation is requested but the processor is not active."""
    pass
