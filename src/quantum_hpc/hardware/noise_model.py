# src/quantum_hpc/hardware/noise_model.py

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class NoiseChannelConfig:
    """
    Configuration for an individual noise channel.

    Attributes:
        channel_type: Type of noise (e.g., 'depolarizing', 'amplitude_damping').
        params: Dictionary containing parameters for that channel (p, gamma, T1, etc.).
        qubits: Optional subset of qubits this noise applies to. If None, applies globally.
        metadata: Arbitrary additional information about the channel (e.g., version, notes).
    """
    channel_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    qubits: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NoiseModelConfig:
    """
    Configuration for an entire noise model, consisting of multiple channels.

    Attributes:
        channels: A list of NoiseChannelConfig objects describing each noise channel.
        combine_mode: How channels should be combined:
                      - 'sequential': Apply one after another in order.
                      - 'parallel': (Reserved for advanced usage where channels act simultaneously.)
    """
    channels: List[NoiseChannelConfig] = field(default_factory=list)
    combine_mode: str = "sequential"

class NoiseModel:
    """
    A high-level noise model that orchestrates multiple noise channels.

    Example usage:
      noise_config = NoiseModelConfig(
          channels=[
              NoiseChannelConfig(
                  channel_type='depolarizing',
                  params={'prob': 0.01}
              ),
              NoiseChannelConfig(
                  channel_type='amplitude_damping',
                  params={'gamma': 0.002},
                  qubits=[0, 1]  # Only apply to qubits 0 and 1
              )
          ]
      )
      noise_model = NoiseModel(noise_config)
    """

    def __init__(self, config: NoiseModelConfig):
        self.config = config
        logger.debug(f"NoiseModel initialized with combine_mode={config.combine_mode}, "
                     f"{len(config.channels)} channels.")

    def apply_noise(self,
                    state_vector: Optional[np.ndarray] = None,
                    density_matrix: Optional[np.ndarray] = None,
                    active_qubits: Optional[List[int]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply all configured noise channels to the provided quantum state.

        Args:
            state_vector: State vector for pure-state simulation (if not None).
            density_matrix: Density matrix for mixed-state simulation (if not None).
            active_qubits: The qubits currently involved in an operation or idle.

        Returns:
            (updated_state_vector, updated_density_matrix)
        """
        sv = state_vector
        dm = density_matrix

        # We only do 'sequential' application in this example.
        for channel_cfg in self.config.channels:
            # Determine which qubits this channel applies to
            noise_qubits = channel_cfg.qubits if channel_cfg.qubits is not None else active_qubits
            # If active_qubits is None, we interpret that as "apply globally" or "no qubits active"
            if noise_qubits is None:
                # If no qubits are specified, skip or treat as global (depending on your design).
                noise_qubits = []

            sv, dm = self._apply_channel(
                channel_cfg.channel_type,
                channel_cfg.params,
                sv,
                dm,
                noise_qubits
            )

        return sv, dm

    def _apply_channel(self,
                       channel_type: str,
                       params: Dict[str, Any],
                       state_vector: Optional[np.ndarray],
                       density_matrix: Optional[np.ndarray],
                       qubits: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Dispatches to the appropriate noise channel implementation.

        Args:
            channel_type: Name of the noise channel (depolarizing, amplitude_damping, etc.).
            params: Dictionary of parameters for that channel (e.g., prob=0.01).
            state_vector: Current state vector or None if using density matrix.
            density_matrix: Current density matrix or None if using state vector.
            qubits: The list of qubit indices to which this noise is applied.

        Returns:
            (updated_state_vector, updated_density_matrix)
        """
        # If no qubits are active, skip applying noise.
        if not qubits:
            return state_vector, density_matrix

        try:
            if channel_type.lower() == 'depolarizing':
                return self._apply_depolarizing(params, state_vector, density_matrix, qubits)
            elif channel_type.lower() == 'amplitude_damping':
                return self._apply_amplitude_damping(params, state_vector, density_matrix, qubits)
            elif channel_type.lower() == 'phase_damping':
                return self._apply_phase_damping(params, state_vector, density_matrix, qubits)
            elif channel_type.lower() == 'thermal':
                return self._apply_thermal_noise(params, state_vector, density_matrix, qubits)
            elif channel_type.lower() == 'crosstalk':
                return self._apply_crosstalk(params, state_vector, density_matrix, qubits)
            else:
                logger.warning(f"Unknown noise channel type: {channel_type}")
                return state_vector, density_matrix
        except Exception as e:
            logger.error(f"Error applying '{channel_type}' noise: {e}")
            return state_vector, density_matrix

    def _apply_depolarizing(self,
                            params: Dict[str, Any],
                            sv: Optional[np.ndarray],
                            dm: Optional[np.ndarray],
                            qubits: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply depolarizing noise channel to the given qubits.
        """
        prob = params.get('prob', 0.0)
        if dm is not None:
            # Density matrix approach
            for _ in qubits:
                dm = (1 - prob) * dm + \
                     (prob / 3) * sum(pauli @ dm @ pauli.conj().T for pauli in [X_GATE, Y_GATE, Z_GATE])
            return sv, dm
        elif sv is not None:
            # State vector approach
            for qubit in qubits:
                if np.random.rand() < prob:
                    error_type = np.random.choice(['X', 'Y', 'Z'])
                    gate_mat = X_GATE if error_type == 'X' else (Y_GATE if error_type == 'Y' else Z_GATE)
                    sv = _apply_gate_to_qubit(gate_mat, sv, qubit)
            return sv, dm
        return sv, dm

    def _apply_amplitude_damping(self,
                                 params: Dict[str, Any],
                                 sv: Optional[np.ndarray],
                                 dm: Optional[np.ndarray],
                                 qubits: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply amplitude damping to the given qubits.
        """
        gamma = params.get('gamma', 0.0)
        if dm is not None:
            # Kraus operators for amplitude damping
            E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
            E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
            for _ in qubits:
                dm = sum(E @ dm @ E.conj().T for E in [E0, E1])
            return sv, dm
        elif sv is not None:
            # Probabilistic decay approach
            for qubit in qubits:
                if np.random.rand() < gamma:
                    sv = _collapse_to_ground(sv, qubit)
            return sv, dm
        return sv, dm

    def _apply_phase_damping(self,
                             params: Dict[str, Any],
                             sv: Optional[np.ndarray],
                             dm: Optional[np.ndarray],
                             qubits: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply phase damping (dephasing) to the given qubits.
        """
        gamma = params.get('gamma', 0.0)
        if dm is not None:
            # Kraus operators
            E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
            E1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
            for _ in qubits:
                dm = sum(E @ dm @ E.conj().T for E in [E0, E1])
            return sv, dm
        elif sv is not None:
            # Probabilistic phase flip approach
            for qubit in qubits:
                if np.random.rand() < gamma:
                    sv = _apply_gate_to_qubit(Z_GATE, sv, qubit)
            return sv, dm
        return sv, dm

    def _apply_thermal_noise(self,
                             params: Dict[str, Any],
                             sv: Optional[np.ndarray],
                             dm: Optional[np.ndarray],
                             qubits: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply a simple thermal noise model, approximating T1/T2 processes.

        Expected parameters: 'temperature', 'coupling'
        """
        temperature = params.get('temperature', 0.0)
        coupling = params.get('coupling', 0.0)
        if temperature <= 0 or coupling <= 0:
            logger.warning("Thermal noise parameters invalid or zero. No effect.")
            return sv, dm

        # Approx: n_thermal = 1/(exp(1/(T*c)) - 1)
        n_thermal = 1 / (np.exp(1 / (temperature * coupling)) - 1)
        gamma_up = coupling * n_thermal
        gamma_down = coupling * (n_thermal + 1)

        if dm is not None:
            # Rudimentary master-equation approach
            L_up = np.array([[0, 1], [0, 0]], dtype=complex)
            L_down = np.array([[0, 0], [1, 0]], dtype=complex)
            for _ in qubits:
                drho = (
                    gamma_up * (L_up @ dm @ L_up.conj().T - 0.5 * (L_up.conj().T @ L_up @ dm + dm @ L_up.conj().T @ L_up))
                    + gamma_down * (L_down @ dm @ L_down.conj().T - 0.5 * (L_down.conj().T @ L_down @ dm + dm @ L_down.conj().T @ L_down))
                )
                dm += drho
            return sv, dm
        elif sv is not None:
            # Probabilistic approach
            for qubit in qubits:
                if np.random.rand() < gamma_up:
                    sv = _excite_qubit(sv, qubit)
                if np.random.rand() < gamma_down:
                    sv = _collapse_to_ground(sv, qubit)
            return sv, dm
        return sv, dm

    def _apply_crosstalk(self,
                         params: Dict[str, Any],
                         sv: Optional[np.ndarray],
                         dm: Optional[np.ndarray],
                         qubits: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply crosstalk effects among active qubits. Typically a ZZ interaction or partial entangling effect.

        Expected params: 'coupling_map' => {(q1, q2): strength, ...}
        """
        coupling_map = params.get('coupling_map', {})
        if not coupling_map:
            return sv, dm

        if dm is not None:
            rho = dm
            for (q1, q2), strength in coupling_map.items():
                # Only apply if either q1 or q2 in the set of qubits
                if q1 in qubits or q2 in qubits:
                    H_int = strength * np.kron(Z_GATE, Z_GATE)
                    # Build full operator if needed. For simplicity, we assume 2-qubit subspace is handled externally
                    # or we do an advanced approach. We'll skip for brevity.
                    # A real approach might do a partial operator or use a stable index mapping.
                    # This demonstration uses a simplified approach or placeholder.
                    pass
            # Return dm unchanged for brevity unless you implement partial U = exp(-iH_int t).
            return sv, dm
        else:
            # State vector approach
            for (q1, q2), strength in coupling_map.items():
                # Only apply if q1 or q2 is in the active set
                if q1 in qubits or q2 in qubits:
                    phase = np.exp(-1j * strength)
                    # apply a naive phase shift to |11>
                    # For a real approach, build an operator or partial entangling gate
                    sv = _apply_two_qubit_phase(sv, q1, q2, phase)
            return sv, dm


#
# Helper Functions + Gate Definitions
#

X_GATE = np.array([[0, 1],
                   [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j],
                   [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0],
                   [0, -1]], dtype=complex)

def _apply_gate_to_qubit(gate: np.ndarray,
                         state: np.ndarray,
                         qubit: int) -> np.ndarray:
    """
    Apply a single-qubit gate to 'qubit' in 'state' (assumed to be a pure state vector).
    """
    n_qubits = int(np.log2(len(state)))
    op = np.eye(1, dtype=complex)
    for i in range(n_qubits):
        op = np.kron(op, gate if i == qubit else np.eye(2, dtype=complex))
    return op @ state

def _apply_two_qubit_phase(state: np.ndarray,
                           q1: int,
                           q2: int,
                           phase: complex) -> np.ndarray:
    """
    Naive approach: multiply the |11> component by 'phase' for qubits q1 and q2.
    """
    new_state = state.copy()
    mask = (1 << q1) | (1 << q2)
    for i in range(len(state)):
        # If qubits q1 and q2 are both '1' in index i, apply phase.
        if (i & mask) == mask:
            new_state[i] *= phase
    return new_state

def _collapse_to_ground(state: np.ndarray, qubit: int) -> np.ndarray:
    """
    Collapse the 'qubit' to |0>. Typically used for amplitude damping or T1 processes.
    """
    new_state = np.zeros_like(state)
    for i in range(len(state)):
        if not (i & (1 << qubit)):
            new_state[i] = state[i]
    norm = np.linalg.norm(new_state)
    if norm > 0:
        new_state /= norm
    return new_state

def _excite_qubit(state: np.ndarray, qubit: int) -> np.ndarray:
    """
    Excite the 'qubit' to |1>. Typically used for thermal noise (gamma_up).
    """
    new_state = np.zeros_like(state)
    mask = 1 << qubit
    for i in range(len(state)):
        if (i & mask) == mask:
            new_state[i] = state[i]
    norm = np.linalg.norm(new_state)
    if norm > 0:
        new_state /= norm
    return new_state
