# src/ionq_benchmarking/core.py

"""
Core IonQ benchmarking modules, including direct randomized benchmarking (DRB),
volumetric benchmarks, and application-level metrics. Uses QuTiP for state
representations and IonQ-like device configurations.
"""

import logging
import numpy as np
import qutip as qt
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """
    Specifies the type of benchmark according to IonQ's methodology:
      - DRB: direct randomized benchmarking
      - VOLUMETRIC: volumetric benchmarks (varying circuit width and depth)
      - APPLICATION: application-oriented benchmarks (Hamiltonian sim, QFT, etc.)
      - COMPONENT: component-level or gate-level benchmarks
    """
    DRB = "direct_randomized_benchmarking"
    VOLUMETRIC = "volumetric"
    APPLICATION = "application_oriented"
    COMPONENT = "component_level"

@dataclass
class GateMetrics:
    """
    Represents gate-level performance metrics derived from IonQ Forte data:
      - error_rate: e.g., in parts per ten thousand (pptt)
      - duration: gate duration in microseconds
      - fidelity: measured or estimated gate fidelity
    """
    error_rate: float
    duration: float
    fidelity: float

class BenchmarkMetrics:
    """
    Tracks global metrics for IonQ-like benchmarks, including DRB error rates
    and application-level circuit fidelity. Also includes a method to calculate
    Hellinger fidelity between measured and ideal distributions.
    """
    def __init__(self):
        # From IonQ paper: typical single-qubit DRB error ~2.0 pptt
        self.single_qubit_error = 0.0020
        # From IonQ paper: typical two-qubit DRB error ~46.4 pptt
        self.two_qubit_error = 0.0464
        self.algorithmic_qubits = 0
        self.circuit_fidelities: Dict[str, float] = {}
        self.hellinger_fidelities: Dict[str, float] = {}

    def calculate_hellinger_fidelity(self,
                                     measured: Dict[str, float],
                                     ideal: Dict[str, float]) -> float:
        """
        Calculate Hellinger fidelity between measured and ideal distributions,
        as IonQ uses in certain comparisons:
          F_c = (Σᵢ sqrt(pᵢ * qᵢ))²

        Args:
            measured: Observed outcome distribution (bitstring -> probability).
            ideal: Ideal or reference distribution (bitstring -> probability).

        Returns:
            Hellinger fidelity in the range [0, 1].
        """
        try:
            fidelity_acc = 0.0
            all_keys = set(measured) | set(ideal)
            for k in all_keys:
                fidelity_acc += np.sqrt(measured.get(k, 0.0) * ideal.get(k, 0.0))
            return fidelity_acc ** 2
        except Exception as e:
            logger.error(f"Hellinger fidelity calculation error: {e}")
            return 0.0

    def update_aq_score(self, circuit_results: Dict[str, float]) -> None:
        """
        Update the 'algorithmic qubit' (#AQ) score based on IonQ's threshold approach:
        if fidelity > 1/e for a circuit of width W, then #AQ >= W.

        Args:
            circuit_results: Mapping of circuit_name -> circuit fidelity.

        The circuit_name is expected to encode the circuit's width, e.g. 'circuit_5'
        for width=5.
        """
        passing_widths = []
        for circuit_name, fidelity in circuit_results.items():
            try:
                # If circuit names are like 'circuit_5', parse out the width
                width_str = circuit_name.split('_')[1]
                width = int(width_str)
                if fidelity > (1 / np.e):
                    passing_widths.append(width)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse width from circuit name: {circuit_name}")
        self.algorithmic_qubits = max(passing_widths) if passing_widths else 0

class IonQDevice:
    """
    Simulates an IonQ Forte-like device with a certain number of qubits and known
    single-/two-qubit gate errors. Uses QuTiP for internal state representation.
    """

    def __init__(self, num_qubits: int = 30):
        """
        Args:
            num_qubits: Number of qubits the IonQ-like device can handle. 
                        Paper suggests Forte system has up to 30 qubits.
        """
        self.num_qubits = num_qubits
        self.metrics = BenchmarkMetrics()
        self.state = qt.tensor([qt.basis(2,0)] * num_qubits)  # Initialize |000..0>
        self.gate_durations = {
            'single': 110.0,   # microseconds for single-qubit gates
            'two_qubit': 900.0 # microseconds for two-qubit gates
        }
        logger.debug(f"IonQDevice initialized with {num_qubits} qubits.")

    def run_drb(self,
                qubits: List[int],
                depth: int,
                p2q: float = 0.25) -> float:
        """
        Run Direct Randomized Benchmarking (DRB) as described in IonQ's paper.
        
        Args:
            qubits: The qubits under test (1 or 2 qubits typically).
            depth: Circuit depth for DRB sequences.
            p2q: Probability of inserting a two-qubit gate at each step 
                 if we have 2 qubits.

        Returns:
            success_prob: Estimated success probability after the DRB sequence.
        """
        success_prob = 1.0
        # For each layer in the DRB sequence
        for _ in range(depth):
            if len(qubits) == 2 and np.random.random() < p2q:
                # Insert a two-qubit gate
                success_prob *= (1.0 - self.metrics.two_qubit_error)
            else:
                # Insert single-qubit gates for each qubit
                for _ in qubits:
                    success_prob *= (1.0 - self.metrics.single_qubit_error)
        logger.info(f"DRB run on qubits={qubits}, depth={depth}, p2q={p2q}, final success={success_prob:.4f}")
        return success_prob

    def measure_probabilities(self) -> Dict[str, float]:
        """
        Measure the current quantum state in the computational basis, returning
        bitstring probabilities above a small threshold (1e-10).
        
        Returns:
            Dictionary mapping bitstring -> probability
        """
        try:
            if self.state.isket:
                # Pure state
                amps = self.state.full().flatten()
                probs = np.abs(amps)**2
            else:
                # Mixed state
                rho = self.state.full()
                probs = np.real(np.diag(rho))

            threshold = 1e-10
            out = {}
            for i, p in enumerate(probs):
                if p > threshold:
                    bitstring = format(i, f'0{self.num_qubits}b')
                    out[bitstring] = p
            return out
        except Exception as e:
            logger.error(f"Error measuring probabilities: {e}")
            return {}

class ApplicationBenchmarks:
    """
    Provides IonQ-like application-oriented benchmarks:
      - Hamiltonian simulation
      - Phase estimation
      - Quantum Fourier Transform
      - Amplitude estimation
      - VQE simulation
      - Monte Carlo sampling
    Each method returns a 'fidelity' or success metric for demonstration.
    """

    def __init__(self, device: IonQDevice):
        """
        Args:
            device: IonQDevice or a mock simulator that we can use to run these benchmarks.
        """
        self.device = device
        self.benchmarks = {
            'hamiltonian_simulation': self.hamiltonian_simulation,
            'phase_estimation': self.phase_estimation,
            'quantum_fourier': self.quantum_fourier,
            'amplitude_estimation': self.amplitude_estimation,
            'vqe_simulation': self.vqe_simulation,
            'monte_carlo': self.monte_carlo
        }
        logger.debug("ApplicationBenchmarks initialized with IonQ-like device.")

    def run_benchmark(self, 
                      name: str, 
                      width: int, 
                      **kwargs) -> float:
        """
        Main entry for application benchmarks. 
        Looks up a benchmark by name and executes it on 'width' qubits.

        Args:
            name: Benchmark name (one of the keys in self.benchmarks).
            width: Number of qubits or circuit width.
            kwargs: Additional parameters for the benchmark function.

        Returns:
            A floating-point 'fidelity' or success score.
        """
        if name in self.benchmarks:
            try:
                logger.info(f"Running IonQ application benchmark '{name}' with width={width}.")
                return self.benchmarks[name](width, **kwargs)
            except Exception as e:
                logger.error(f"Benchmark '{name}' failed: {e}")
                return 0.0
        logger.warning(f"Benchmark '{name}' not recognized.")
        return 0.0

    def hamiltonian_simulation(self, width: int, **kwargs) -> float:
        """
        Example Hamiltonian simulation. In real code, you'd build a Hamiltonian
        (e.g., a Heisenberg model) for 'width' qubits, evolve it, measure fidelity.

        Args:
            width: Number of qubits

        Returns:
            A placeholder success or fidelity score.
        """
        logger.debug(f"Hamiltonian simulation with width={width}.")
        # Placeholder
        return np.random.uniform(0.9, 1.0)

    def phase_estimation(self, width: int, **kwargs) -> float:
        logger.debug(f"Phase estimation with width={width}.")
        return np.random.uniform(0.85, 1.0)

    def quantum_fourier(self, width: int, **kwargs) -> float:
        logger.debug(f"Quantum Fourier Transform with width={width}.")
        return np.random.uniform(0.88, 1.0)

    def amplitude_estimation(self, width: int, **kwargs) -> float:
        logger.debug(f"Amplitude estimation with width={width}.")
        return np.random.uniform(0.9, 1.0)

    def vqe_simulation(self, width: int, **kwargs) -> float:
        logger.debug(f"VQE simulation with width={width}.")
        return np.random.uniform(0.87, 1.0)

    def monte_carlo(self, width: int, **kwargs) -> float:
        logger.debug(f"Monte Carlo sampling with width={width}.")
        return np.random.uniform(0.88, 1.0)
