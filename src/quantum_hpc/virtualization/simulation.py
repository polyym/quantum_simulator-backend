# src/quantum_hpc/virtualization/simulation.py

import logging
from typing import Optional, List, Dict, Any, Union

import numpy as np
from dataclasses import dataclass, field

from src.quantum_hpc.hardware.noise_model import NoiseModel
from src.quantum_hpc.hardware.topology import QuantumTopology

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """
    Configuration parameters for the simulation engine.

    Attributes:
        num_qubits: Number of qubits to simulate.
        initial_state: Optional initial state vector or density matrix.
        use_density_matrix: Flag to indicate whether we're simulating a mixed-state (True) or pure-state (False).
        apply_noise: If True, apply noise channels at each step (gates, idle, measurement).
        noise_config: Optional NoiseModel that describes the channels to apply.
        topology: Optional QuantumTopology describing connectivity constraints or distances.
        metadata: Arbitrary dictionary for additional simulation parameters or environment details.
    """
    num_qubits: int
    initial_state: Optional[Union[np.ndarray, List[complex]]] = None
    use_density_matrix: bool = False
    apply_noise: bool = True
    noise_config: Optional[NoiseModel] = None
    topology: Optional[QuantumTopology] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumSimulationEngine:
    """
    Software-based quantum simulation engine. This class manages:
      1. State representation (vector or density matrix).
      2. Gate application (single and multi-qubit).
      3. Noise injection if configured (via NoiseModel).
      4. Measurement in various bases (Z, X, Y).
      5. (Optional) HPC-level partitioning or distribution of states (not fully implemented here).

    You can extend this to support advanced HPC frameworks, GPU acceleration, or
    library integrations (e.g., QuTiP, Cirq).
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation engine using the provided config.

        Args:
            config: SimulationConfig with details about qubit count, initial state, noise, etc.
        """
        self.config = config
        self.num_qubits = config.num_qubits
        self.use_density_matrix = config.use_density_matrix
        self.state_vector: Optional[np.ndarray] = None
        self.density_matrix: Optional[np.ndarray] = None

        # Initialize state (vector or density matrix)
        self._initialize_state()

        # If we have a topology, we can reference it when validating two-qubit gates, etc.
        self.topology = config.topology

        # Store or create noise model
        self.noise_model = config.noise_config
        logger.debug(f"QuantumSimulationEngine initialized with {self.num_qubits} qubits.")

    def _initialize_state(self) -> None:
        """
        Set up the initial quantum state from config.
        """
        if self.use_density_matrix:
            # If user provided a density matrix, load it. Otherwise, build from a default pure state.
            if self.config.initial_state is not None:
                dm = np.array(self.config.initial_state, dtype=complex)
                if dm.shape != (2**self.num_qubits, 2**self.num_qubits):
                    raise ValueError("Invalid density matrix dimensions.")
                self.density_matrix = dm
                self.state_vector = None
            else:
                # Default to |0...0><0...0|
                self.density_matrix = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)
                self.density_matrix[0, 0] = 1.0
                self.state_vector = None
        else:
            # Pure state simulation
            if self.config.initial_state is not None:
                sv = np.array(self.config.initial_state, dtype=complex)
                if sv.shape != (2**self.num_qubits,):
                    raise ValueError("Invalid state vector length.")
                self.state_vector = sv
                self.density_matrix = None
            else:
                # Default to |0...0>
                self.state_vector = np.zeros(2**self.num_qubits, dtype=complex)
                self.state_vector[0] = 1.0
                self.density_matrix = None

    def apply_gate(self, 
                   gate_matrix: np.ndarray,
                   qubits: List[int],
                   gate_name: str = "custom_gate") -> None:
        """
        Apply a multi-qubit gate (or single-qubit gate) to the simulation state.

        Args:
            gate_matrix: The operator in matrix form for the targeted qubits (in ascending order).
            qubits: The list of qubit indices the gate acts on, e.g. [0] or [1, 2].
            gate_name: Optional name for logging or reference.
        """
        # Validate qubit indices with topology if present
        if self.topology and len(qubits) > 1:
            # Check if all qubits are connected in topology
            if not self._check_connectivity(qubits):
                logger.warning(f"Gate '{gate_name}' on qubits {qubits} not allowed by topology.")
                return

        # Apply the gate to the state
        if self.state_vector is not None:
            logger.debug(f"Applying {gate_name} on qubits={qubits} in state_vector mode.")
            self.state_vector = self._apply_operator_to_state(gate_matrix, qubits, self.state_vector)
            if self.config.apply_noise and self.noise_model:
                self.state_vector, _ = self.noise_model.apply_noise(
                    state_vector=self.state_vector,
                    active_qubits=qubits
                )
        elif self.density_matrix is not None:
            logger.debug(f"Applying {gate_name} on qubits={qubits} in density_matrix mode.")
            self.density_matrix = self._apply_operator_to_density(gate_matrix, qubits, self.density_matrix)
            if self.config.apply_noise and self.noise_model:
                _, self.density_matrix = self.noise_model.apply_noise(
                    density_matrix=self.density_matrix,
                    active_qubits=qubits
                )

    def measure(self, 
                qubits: List[int], 
                basis: str = "Z") -> Tuple[List[int], float]:
        """
        Measure the specified qubits in the given basis. Returns measurement outcomes and a naive fidelity or confidence.

        Args:
            qubits: The qubits to measure.
            basis: "Z", "X", or "Y".

        Returns:
            (List of 0/1 results, float approximate measurement fidelity).
        """
        # Convert measure basis to an operator or rotate state if needed
        if basis not in ["Z", "X", "Y"]:
            logger.warning(f"Unknown measurement basis '{basis}'. Defaulting to Z.")
            basis = "Z"

        # If we have noise, we can apply measurement errors
        if self.config.apply_noise and self.noise_model:
            if self.state_vector is not None:
                self.state_vector, _ = self.noise_model.apply_noise(
                    state_vector=self.state_vector,
                    active_qubits=qubits
                )
            elif self.density_matrix is not None:
                _, self.density_matrix = self.noise_model.apply_noise(
                    density_matrix=self.density_matrix,
                    active_qubits=qubits
                )

        # We'll just do a naive projective measurement approach
        if self.state_vector is not None:
            outcomes = self._measure_state_vector(qubits, basis)
            fidelity = 0.99  # A placeholder for real measurement fidelity
            return outcomes, fidelity
        elif self.density_matrix is not None:
            outcomes = self._measure_density_matrix(qubits, basis)
            fidelity = 0.99
            return outcomes, fidelity
        else:
            logger.error("No valid state to measure.")
            return [], 0.0

    #
    # Internal State Manipulation Helpers
    #
    def _apply_operator_to_state(self, 
                                 operator: np.ndarray, 
                                 qubits: List[int], 
                                 state_vec: np.ndarray) -> np.ndarray:
        """
        Apply a multi-qubit operator to a pure state vector. 
        The operator must match the dimension 2^(len(qubits)).
        """
        n_qubits = self.num_qubits
        # Build full operator by tensoring identity for unaffected qubits
        full_op = self._embed_operator(operator, qubits, n_qubits)
        return full_op @ state_vec

    def _apply_operator_to_density(self, 
                                   operator: np.ndarray, 
                                   qubits: List[int],
                                   density_mat: np.ndarray) -> np.ndarray:
        """
        Apply a multi-qubit operator to a density matrix:
        ρ -> UρU†
        """
        n_qubits = self.num_qubits
        full_op = self._embed_operator(operator, qubits, n_qubits)
        return full_op @ density_mat @ full_op.conjugate().T

    def _embed_operator(self, 
                        local_op: np.ndarray, 
                        target_qubits: List[int],
                        total_qubits: int) -> np.ndarray:
        """
        Construct a full 2^n x 2^n operator from 'local_op' which acts on 'target_qubits'
        and leaves others as identity.
        """
        # Sort target qubits to match typical ordering
        sorted_qubits = sorted(target_qubits)
        n_target = len(sorted_qubits)
        # Check dimension
        if local_op.shape != (2**n_target, 2**n_target):
            raise ValueError(
                f"Operator dimension {local_op.shape} doesn't match len(target_qubits)={n_target}."
            )

        # We'll build step by step with Kron( identity or local_op ), for each qubit in [0..n-1].
        op = np.array([[1.0]], dtype=complex)
        idx = 0
        for qubit_index in range(total_qubits):
            if qubit_index in sorted_qubits:
                # This qubit is part of local_op
                sub_dim = 2
                op = np.kron(op, self._extract_sub_operator(local_op, idx, n_target))
                idx += 1
            else:
                # This qubit is unaffected => identity
                op = np.kron(op, np.eye(2, dtype=complex))
        return op

    def _extract_sub_operator(self, 
                              local_op: np.ndarray,
                              idx: int,
                              n_target: int) -> np.ndarray:
        """
        Not strictly needed if we apply local_op in a single chunk. 
        But for clarity, we assume each qubit dimension is 2 
        and we sequentially pick the appropriate subspace of local_op.
        In many frameworks, you apply local_op in one shot when you 
        hit the first target qubit in the iteration.
        """
        # For a multi-qubit gate, we only apply the entire local_op once 
        # when we reach the first qubit in sorted_qubits, 
        # and identity for the subsequent 'n_target-1' placeholders. 
        # However, the embedding approach used can differ.

        # Simplified approach: if idx == 0, return local_op. Otherwise, return identity(2).
        if idx == 0:
            return local_op
        else:
            # We embed local_op at the earliest qubit, and the rest are identity.
            # This is one of multiple ways to embed. Another approach is building the entire op 
            # for all target qubits in one shot, skipping this function.
            return np.eye(2, dtype=complex)

    #
    # Measurement Routines
    #
    def _measure_state_vector(self,
                              qubits: List[int],
                              basis: str) -> List[int]:
        """
        Projective measurement of 'qubits' in given basis from a pure state vector.
        We'll do a naive approach: if basis != Z, rotate the state accordingly, 
        then measure in the Z basis.
        """
        # If basis != Z, rotate state (e.g., apply H for X, S†H for Y, etc.)
        rotated_state = self._rotate_state_for_measurement(self.state_vector, qubits, basis)
        
        # Now measure in the Z basis
        outcomes = []
        for q in qubits:
            # We measure qubit q by partial trace approach
            outcome = self._sample_measurement_z(rotated_state, q)
            outcomes.append(outcome)
            # Then we collapse the wavefunction for subsequent qubits
            rotated_state = self._collapse_wavefunction_z(rotated_state, q, outcome)
        # After measuring, we store the final collapsed state back
        self.state_vector = rotated_state
        return outcomes

    def _measure_density_matrix(self,
                                qubits: List[int],
                                basis: str) -> List[int]:
        """
        Projective measurement from a density matrix. 
        Similarly, rotate if not measuring in Z, then do a partial trace approach.
        """
        rotated_dm = self._rotate_density_for_measurement(self.density_matrix, qubits, basis)
        outcomes = []
        for q in qubits:
            outcome = self._sample_measurement_z_dm(rotated_dm, q)
            outcomes.append(outcome)
            rotated_dm = self._collapse_density_z(rotated_dm, q, outcome)
        self.density_matrix = rotated_dm
        return outcomes

    #
    # Additional Helpers
    #
    def _check_connectivity(self, qubits: List[int]) -> bool:
        """
        If a topology is defined, verify that multi-qubit gates are valid for that adjacency.
        """
        if len(qubits) < 2 or not self.topology:
            return True
        # For a 2-qubit gate, check if qubits[1] is in neighbors of qubits[0], etc.
        # For a multi-qubit gate, you'd do more complex checking or assume partial order.
        for i in range(len(qubits) - 1):
            a, b = qubits[i], qubits[i+1]
            if b not in self.topology.get_neighbors(a):
                return False
        return True

    def _rotate_state_for_measurement(self, 
                                      state_vec: np.ndarray,
                                      qubits: List[int],
                                      basis: str) -> np.ndarray:
        """
        Apply rotation to measure in X or Y basis. 
        For demonstration:
          - X basis => apply H on each measured qubit
          - Y basis => apply S† then H
        """
        rotated = state_vec
        for q in qubits:
            if basis == "X":
                rotated = self._apply_operator_to_state(_H_GATE, [q], rotated)
            elif basis == "Y":
                rotated = self._apply_operator_to_state(_S_DAGGER_GATE, [q], rotated)
                rotated = self._apply_operator_to_state(_H_GATE, [q], rotated)
        return rotated

    def _rotate_density_for_measurement(self,
                                        dm: np.ndarray,
                                        qubits: List[int],
                                        basis: str) -> np.ndarray:
        """
        Same logic as _rotate_state_for_measurement but for density matrices.
        """
        rotated = dm
        for q in qubits:
            if basis == "X":
                rotated = self._apply_operator_to_density(_H_GATE, [q], rotated)
            elif basis == "Y":
                rotated = self._apply_operator_to_density(_S_DAGGER_GATE, [q], rotated)
                rotated = self._apply_operator_to_density(_H_GATE, [q], rotated)
        return rotated

    def _sample_measurement_z(self, state_vec: np.ndarray, qubit: int) -> int:
        """
        Compute probability of 0 or 1 for qubit, then sample a random outcome.
        """
        prob_0 = 0.0
        n_qubits = self.num_qubits
        mask = 1 << qubit
        # Probability of qubit=0 => sum of |amplitudes|^2 of basis states with that bit=0
        for i in range(len(state_vec)):
            if (i & mask) == 0:
                prob_0 += np.abs(state_vec[i])**2
        outcome = 0 if np.random.rand() < prob_0 else 1
        return outcome

    def _collapse_wavefunction_z(self, state_vec: np.ndarray, qubit: int, outcome: int) -> np.ndarray:
        """
        Collapse wavefunction after measuring qubit in Z basis with given outcome (0 or 1).
        """
        new_state = state_vec.copy()
        mask = 1 << qubit
        for i in range(len(new_state)):
            bit = 1 if (i & mask) else 0
            if bit != outcome:
                new_state[i] = 0.0
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state /= norm
        return new_state

    def _sample_measurement_z_dm(self, dm: np.ndarray, qubit: int) -> int:
        """
        For a density matrix, probability of measuring qubit=0 => partial trace over subspace.
        This is simplified to a direct approach: sum diagonal elements of subspace.
        """
        prob_0 = 0.0
        dim = 2**self.num_qubits
        mask = 1 << qubit
        for i in range(dim):
            bit = 1 if (i & mask) else 0
            if bit == 0:
                prob_0 += np.real(dm[i, i])
        outcome = 0 if np.random.rand() < prob_0 else 1
        return outcome

    def _collapse_density_z(self, dm: np.ndarray, qubit: int, outcome: int) -> np.ndarray:
        """
        Collapse density matrix after measuring qubit in Z basis => project onto |outcome>.
        """
        dim = 2**self.num_qubits
        projector = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            bit = 1 if (i & (1 << qubit)) else 0
            if bit == outcome:
                projector[i, i] = 1.0
        # ρ -> P ρ P / Tr(P ρ P)
        new_dm = projector @ dm @ projector
        trace_val = np.real(np.trace(new_dm))
        if trace_val > 0:
            new_dm /= trace_val
        return new_dm


#
# Common gates for measurement rotations
#
_H_GATE = (1.0 / np.sqrt(2)) * np.array([[1, 1],
                                        [1, -1]], dtype=complex)

_S_DAGGER_GATE = np.array([[1, 0],
                           [0, 0 - 1j]], dtype=complex)
