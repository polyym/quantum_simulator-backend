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
        Apply thermal noise model based on physical T1/T2 processes.

        This implements a physically accurate thermal relaxation model using
        the Lindblad master equation formalism.

        Physics Background:
        - Thermal equilibrium is determined by the Boltzmann distribution
        - The thermal occupation number follows Bose-Einstein statistics:
          n_th = 1 / (exp(ℏω / k_B T) - 1)
        - Transition rates depend on n_th:
          γ↑ = γ * n_th (excitation rate)
          γ↓ = γ * (n_th + 1) (relaxation rate)

        Expected parameters:
        - 'temperature_kelvin': Temperature in Kelvin (default: 0.02 K for dilution fridge)
        - 'frequency_ghz': Qubit transition frequency in GHz (default: 5.0 GHz)
        - 'T1_us': T1 relaxation time in microseconds (default: 100 μs)
        - 'gate_time_us': Duration of the operation in microseconds (default: 0.1 μs)

        Alternative legacy parameters (for backward compatibility):
        - 'temperature': Dimensionless temperature parameter
        - 'coupling': Dimensionless coupling parameter
        """
        # Physical constants
        HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
        KB = 1.380649e-23       # Boltzmann constant (J/K)

        # Try to get physical parameters first
        temperature_kelvin = params.get('temperature_kelvin', None)
        frequency_ghz = params.get('frequency_ghz', 5.0)
        T1_us = params.get('T1_us', 100.0)
        gate_time_us = params.get('gate_time_us', 0.1)

        if temperature_kelvin is not None:
            # Use physical units
            if temperature_kelvin <= 0:
                logger.warning("Temperature must be positive. No thermal noise applied.")
                return sv, dm

            # Convert units
            omega = 2 * np.pi * frequency_ghz * 1e9  # Angular frequency (rad/s)
            T1 = T1_us * 1e-6  # T1 in seconds
            gate_time = gate_time_us * 1e-6  # Gate time in seconds

            # Calculate thermal occupation number using Bose-Einstein statistics
            # n_th = 1 / (exp(ℏω / k_B T) - 1)
            exponent = (HBAR * omega) / (KB * temperature_kelvin)

            if exponent > 700:  # Prevent overflow
                n_thermal = 0.0  # Effectively zero temperature
            else:
                n_thermal = 1.0 / (np.exp(exponent) - 1)

            # Base relaxation rate from T1
            gamma_base = 1.0 / T1

            # Transition rates (per second)
            # γ↓ = γ * (n_th + 1) - stimulated + spontaneous emission
            # γ↑ = γ * n_th - absorption from thermal bath
            gamma_down = gamma_base * (n_thermal + 1)
            gamma_up = gamma_base * n_thermal

            # Convert to probability for this gate duration
            p_down = 1 - np.exp(-gamma_down * gate_time)
            p_up = 1 - np.exp(-gamma_up * gate_time)

            logger.debug(
                f"Thermal noise: T={temperature_kelvin:.3f}K, f={frequency_ghz:.1f}GHz, "
                f"n_th={n_thermal:.2e}, p_down={p_down:.2e}, p_up={p_up:.2e}"
            )

        else:
            # Legacy dimensionless parameters (backward compatibility)
            temperature = params.get('temperature', 0.0)
            coupling = params.get('coupling', 0.0)

            if temperature <= 0 or coupling <= 0:
                logger.warning("Thermal noise parameters invalid or zero. No effect.")
                return sv, dm

            # Legacy formula (dimensionless approximation)
            # Note: This is not physically accurate but preserved for compatibility
            try:
                exponent = 1.0 / (temperature * coupling)
                if exponent > 700:
                    n_thermal = 0.0
                else:
                    n_thermal = 1.0 / (np.exp(exponent) - 1)
            except (ZeroDivisionError, OverflowError):
                n_thermal = 0.0

            gamma_up = coupling * n_thermal
            gamma_down = coupling * (n_thermal + 1)
            p_up = gamma_up
            p_down = gamma_down

            logger.debug(f"Thermal noise (legacy): n_th={n_thermal:.2e}")

        if dm is not None:
            # Lindblad master equation for thermal relaxation
            # dρ/dt = γ↓ D[σ_-](ρ) + γ↑ D[σ_+](ρ)
            # where D[L](ρ) = L ρ L† - 1/2 {L† L, ρ}
            L_down = np.array([[0, 1], [0, 0]], dtype=complex)  # σ_- (lowering)
            L_up = np.array([[0, 0], [1, 0]], dtype=complex)    # σ_+ (raising)

            for _ in qubits:
                # Relaxation (|1⟩ → |0⟩)
                drho_down = p_down * (
                    L_down @ dm @ L_down.conj().T
                    - 0.5 * (L_down.conj().T @ L_down @ dm + dm @ L_down.conj().T @ L_down)
                )
                # Excitation (|0⟩ → |1⟩)
                drho_up = p_up * (
                    L_up @ dm @ L_up.conj().T
                    - 0.5 * (L_up.conj().T @ L_up @ dm + dm @ L_up.conj().T @ L_up)
                )
                dm = dm + drho_down + drho_up

            return sv, dm

        elif sv is not None:
            # Probabilistic Monte Carlo approach for state vector
            for qubit in qubits:
                if np.random.rand() < p_up:
                    sv = _excite_qubit(sv, qubit)
                if np.random.rand() < p_down:
                    sv = _collapse_to_ground(sv, qubit)
            return sv, dm

        return sv, dm

    def _apply_crosstalk(self,
                         params: Dict[str, Any],
                         sv: Optional[np.ndarray],
                         dm: Optional[np.ndarray],
                         qubits: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply crosstalk effects among active qubits using ZZ interaction model.

        Physics Background:
        Crosstalk in superconducting qubits arises from residual ZZ coupling between
        qubits. The effective Hamiltonian is:
            H_crosstalk = Σ_{i,j} ζ_{ij} Z_i ⊗ Z_j

        where ζ_{ij} is the coupling strength (typically 10-100 kHz for adjacent qubits).

        The time evolution operator is:
            U = exp(-i H t) = exp(-i ζ t Z_i ⊗ Z_j)

        This results in conditional phase accumulation on the |11⟩ state.

        Expected params:
            'coupling_map': Dict mapping (q1, q2) tuples to coupling strength (radians)
            'gate_time': Optional gate duration for time-dependent coupling

        References:
            - Krantz et al., "A Quantum Engineer's Guide to Superconducting Qubits"
            - Sheldon et al., "Procedure for systematically tuning up cross-talk"
        """
        coupling_map = params.get('coupling_map', {})
        gate_time = params.get('gate_time', 1.0)  # Normalized time unit

        if not coupling_map:
            return sv, dm

        n_qubits = int(np.log2(len(sv if sv is not None else dm)))

        if dm is not None:
            # Density matrix evolution: ρ → U ρ U†
            for (q1, q2), strength in coupling_map.items():
                if q1 >= n_qubits or q2 >= n_qubits:
                    continue
                # Only apply if either q1 or q2 is in the active set
                if q1 in qubits or q2 in qubits:
                    # Build the ZZ unitary: exp(-i ζ t Z⊗Z)
                    # ZZ has eigenvalues ±1, so exp(-i ζ t ZZ) = cos(ζt)I - i sin(ζt)ZZ
                    zz_unitary = _build_zz_unitary(n_qubits, q1, q2, strength * gate_time)
                    dm = zz_unitary @ dm @ zz_unitary.conj().T

            return sv, dm
        elif sv is not None:
            # State vector evolution: |ψ⟩ → U|ψ⟩
            for (q1, q2), strength in coupling_map.items():
                if q1 >= n_qubits or q2 >= n_qubits:
                    continue
                # Only apply if q1 or q2 is in the active set
                if q1 in qubits or q2 in qubits:
                    # Apply ZZ phase: |00⟩→|00⟩, |01⟩→e^{iζt}|01⟩,
                    #                 |10⟩→e^{iζt}|10⟩, |11⟩→|11⟩
                    # This is the diagonal of exp(-i ζ t ZZ)
                    sv = _apply_zz_phase(sv, q1, q2, strength * gate_time)

            return sv, dm

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


def _apply_zz_phase(state: np.ndarray,
                    q1: int,
                    q2: int,
                    angle: float) -> np.ndarray:
    """
    Apply ZZ interaction phase to a state vector.

    The ZZ operator has eigenvalues:
        |00⟩: Z⊗Z = (+1)(+1) = +1  → phase e^{-iθ}
        |01⟩: Z⊗Z = (+1)(-1) = -1  → phase e^{+iθ}
        |10⟩: Z⊗Z = (-1)(+1) = -1  → phase e^{+iθ}
        |11⟩: Z⊗Z = (-1)(-1) = +1  → phase e^{-iθ}

    Args:
        state: State vector
        q1, q2: Qubit indices
        angle: Rotation angle (ζ * t)

    Returns:
        State vector with ZZ phase applied
    """
    new_state = state.copy()
    mask_q1 = 1 << q1
    mask_q2 = 1 << q2

    phase_plus = np.exp(-1j * angle)   # For eigenvalue +1 (|00⟩, |11⟩)
    phase_minus = np.exp(1j * angle)   # For eigenvalue -1 (|01⟩, |10⟩)

    for i in range(len(state)):
        bit_q1 = (i & mask_q1) >> q1
        bit_q2 = (i & mask_q2) >> q2
        # ZZ eigenvalue is (-1)^(bit_q1 XOR bit_q2) * ... wait, let's be careful:
        # Z|0⟩ = +|0⟩, Z|1⟩ = -|1⟩
        # So ZZ|b1,b2⟩ = (-1)^b1 * (-1)^b2 |b1,b2⟩ = (-1)^(b1+b2) |b1,b2⟩
        eigenvalue_sign = (bit_q1 + bit_q2) % 2  # 0 for |00⟩,|11⟩; 1 for |01⟩,|10⟩
        if eigenvalue_sign == 0:
            new_state[i] *= phase_plus
        else:
            new_state[i] *= phase_minus

    return new_state


def _build_zz_unitary(n_qubits: int,
                      q1: int,
                      q2: int,
                      angle: float) -> np.ndarray:
    """
    Build the full ZZ unitary matrix for density matrix evolution.

    U = exp(-i θ Z_q1 ⊗ Z_q2)

    This is a diagonal matrix with entries e^{-iθ(±1)} depending on
    the ZZ eigenvalue of each basis state.

    Args:
        n_qubits: Total number of qubits
        q1, q2: Qubit indices for the ZZ interaction
        angle: Rotation angle (ζ * t)

    Returns:
        2^n × 2^n unitary matrix
    """
    dim = 2 ** n_qubits
    unitary = np.zeros((dim, dim), dtype=complex)

    mask_q1 = 1 << q1
    mask_q2 = 1 << q2

    phase_plus = np.exp(-1j * angle)
    phase_minus = np.exp(1j * angle)

    for i in range(dim):
        bit_q1 = (i & mask_q1) >> q1
        bit_q2 = (i & mask_q2) >> q2
        eigenvalue_sign = (bit_q1 + bit_q2) % 2
        if eigenvalue_sign == 0:
            unitary[i, i] = phase_plus
        else:
            unitary[i, i] = phase_minus

    return unitary

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
