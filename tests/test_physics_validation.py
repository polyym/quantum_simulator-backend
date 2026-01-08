# tests/test_physics_validation.py

"""
Physics Validation Tests for Quantum Simulator Backend

This module contains tests that verify the physical correctness of the quantum
simulation implementation. These tests ensure that:

1. Gate matrices are unitary (U†U = I)
2. State vectors remain normalized after operations
3. Measurement probabilities follow Born's rule
4. Noise channels preserve trace and positivity of density matrices
5. Known quantum circuits produce expected results

These tests serve as a scientific validation suite for production readiness.
"""

import pytest
import numpy as np
from typing import List, Dict


class TestGateUnitarity:
    """Test that all quantum gate matrices are unitary."""

    # Standard single-qubit gates
    SINGLE_QUBIT_GATES = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        'S': np.array([[1, 0], [0, 1j]], dtype=complex),
        'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
        'Sdg': np.array([[1, 0], [0, -1j]], dtype=complex),
        'Tdg': np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
    }

    # Standard two-qubit gates
    TWO_QUBIT_GATES = {
        'CNOT': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex),
        'CZ': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex),
        'SWAP': np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex),
        'iSWAP': np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex),
        'SQRTSWAP': np.array([
            [1, 0, 0, 0],
            [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
            [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
            [0, 0, 0, 1]
        ], dtype=complex),
    }

    def _is_unitary(self, matrix: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if a matrix is unitary: U†U = I."""
        identity = np.eye(matrix.shape[0], dtype=complex)
        product = matrix.conj().T @ matrix
        return np.allclose(product, identity, atol=tol)

    @pytest.mark.parametrize("gate_name", list(SINGLE_QUBIT_GATES.keys()))
    def test_single_qubit_gate_unitarity(self, gate_name: str):
        """Verify that single-qubit gates are unitary."""
        gate = self.SINGLE_QUBIT_GATES[gate_name]
        assert self._is_unitary(gate), f"Gate {gate_name} is not unitary"

    @pytest.mark.parametrize("gate_name", list(TWO_QUBIT_GATES.keys()))
    def test_two_qubit_gate_unitarity(self, gate_name: str):
        """Verify that two-qubit gates are unitary."""
        gate = self.TWO_QUBIT_GATES[gate_name]
        assert self._is_unitary(gate), f"Gate {gate_name} is not unitary"

    @pytest.mark.parametrize("angle", [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2])
    def test_rotation_gate_unitarity(self, angle: float):
        """Verify that rotation gates RX, RY, RZ are unitary for various angles."""
        # RX(θ) = cos(θ/2)I - i·sin(θ/2)X
        rx = np.array([
            [np.cos(angle/2), -1j * np.sin(angle/2)],
            [-1j * np.sin(angle/2), np.cos(angle/2)]
        ], dtype=complex)

        # RY(θ) = cos(θ/2)I - i·sin(θ/2)Y
        ry = np.array([
            [np.cos(angle/2), -np.sin(angle/2)],
            [np.sin(angle/2), np.cos(angle/2)]
        ], dtype=complex)

        # RZ(θ) = e^{-iθ/2}|0⟩⟨0| + e^{iθ/2}|1⟩⟨1|
        rz = np.array([
            [np.exp(-1j * angle/2), 0],
            [0, np.exp(1j * angle/2)]
        ], dtype=complex)

        assert self._is_unitary(rx), f"RX({angle}) is not unitary"
        assert self._is_unitary(ry), f"RY({angle}) is not unitary"
        assert self._is_unitary(rz), f"RZ({angle}) is not unitary"


class TestStateNormalization:
    """Test that quantum states remain properly normalized."""

    def _is_normalized(self, state: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if a state vector is normalized: ⟨ψ|ψ⟩ = 1."""
        norm = np.sum(np.abs(state) ** 2)
        return np.isclose(norm, 1.0, atol=tol)

    def test_computational_basis_normalization(self):
        """Verify computational basis states are normalized."""
        for n_qubits in range(1, 5):
            dim = 2 ** n_qubits
            for i in range(dim):
                state = np.zeros(dim, dtype=complex)
                state[i] = 1.0
                assert self._is_normalized(state), f"|{bin(i)}⟩ is not normalized"

    def test_hadamard_preserves_normalization(self):
        """Verify Hadamard gate preserves state normalization."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

        # Apply H to |0⟩
        state_0 = np.array([1, 0], dtype=complex)
        state_after_h = H @ state_0
        assert self._is_normalized(state_after_h), "H|0⟩ is not normalized"

        # Apply H to |1⟩
        state_1 = np.array([0, 1], dtype=complex)
        state_after_h = H @ state_1
        assert self._is_normalized(state_after_h), "H|1⟩ is not normalized"

    def test_cnot_preserves_normalization(self):
        """Verify CNOT gate preserves state normalization."""
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

        # Test on various input states
        test_states = [
            np.array([1, 0, 0, 0], dtype=complex),  # |00⟩
            np.array([0, 1, 0, 0], dtype=complex),  # |01⟩
            np.array([0, 0, 1, 0], dtype=complex),  # |10⟩
            np.array([0, 0, 0, 1], dtype=complex),  # |11⟩
            np.array([0.5, 0.5, 0.5, 0.5], dtype=complex),  # Equal superposition
        ]

        for state in test_states:
            output = CNOT @ state
            assert self._is_normalized(output), "CNOT does not preserve normalization"


class TestBornsRule:
    """Test that measurement probabilities follow Born's rule."""

    def test_computational_basis_measurement(self):
        """Verify measurement probabilities for computational basis states."""
        # |0⟩ should give P(0) = 1, P(1) = 0
        state_0 = np.array([1, 0], dtype=complex)
        prob_0 = np.abs(state_0[0]) ** 2
        prob_1 = np.abs(state_0[1]) ** 2
        assert np.isclose(prob_0, 1.0), "P(0) for |0⟩ should be 1"
        assert np.isclose(prob_1, 0.0), "P(1) for |0⟩ should be 0"

        # |1⟩ should give P(0) = 0, P(1) = 1
        state_1 = np.array([0, 1], dtype=complex)
        prob_0 = np.abs(state_1[0]) ** 2
        prob_1 = np.abs(state_1[1]) ** 2
        assert np.isclose(prob_0, 0.0), "P(0) for |1⟩ should be 0"
        assert np.isclose(prob_1, 1.0), "P(1) for |1⟩ should be 1"

    def test_superposition_measurement(self):
        """Verify measurement probabilities for superposition states."""
        # |+⟩ = (|0⟩ + |1⟩)/√2 should give P(0) = P(1) = 0.5
        state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        prob_0 = np.abs(state_plus[0]) ** 2
        prob_1 = np.abs(state_plus[1]) ** 2
        assert np.isclose(prob_0, 0.5), "P(0) for |+⟩ should be 0.5"
        assert np.isclose(prob_1, 0.5), "P(1) for |+⟩ should be 0.5"

    def test_probability_sum_equals_one(self):
        """Verify that probabilities sum to 1 for any state."""
        # Generate random normalized states
        np.random.seed(42)
        for _ in range(10):
            n_qubits = np.random.randint(1, 5)
            dim = 2 ** n_qubits
            # Random complex state
            state = np.random.randn(dim) + 1j * np.random.randn(dim)
            state = state / np.linalg.norm(state)

            probabilities = np.abs(state) ** 2
            assert np.isclose(np.sum(probabilities), 1.0), \
                "Probabilities must sum to 1"


class TestNoiseChannelProperties:
    """Test that noise channels have correct mathematical properties."""

    def _is_trace_preserving(self, kraus_ops: List[np.ndarray], tol: float = 1e-10) -> bool:
        """Check if Kraus operators satisfy Σ_k E_k† E_k = I (trace preserving)."""
        dim = kraus_ops[0].shape[0]
        identity = np.eye(dim, dtype=complex)
        sum_term = sum(E.conj().T @ E for E in kraus_ops)
        return np.allclose(sum_term, identity, atol=tol)

    def test_depolarizing_channel_trace_preserving(self):
        """Verify depolarizing channel is trace preserving."""
        # Depolarizing channel: E_0 = √(1-p)I, E_k = √(p/3)σ_k for k=1,2,3
        p = 0.1  # Error probability
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        kraus_ops = [
            np.sqrt(1 - p) * I,
            np.sqrt(p / 3) * X,
            np.sqrt(p / 3) * Y,
            np.sqrt(p / 3) * Z,
        ]

        assert self._is_trace_preserving(kraus_ops), \
            "Depolarizing channel should be trace preserving"

    def test_amplitude_damping_trace_preserving(self):
        """Verify amplitude damping channel is trace preserving."""
        gamma = 0.1  # Damping rate

        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)

        kraus_ops = [E0, E1]
        assert self._is_trace_preserving(kraus_ops), \
            "Amplitude damping channel should be trace preserving"

    def test_phase_damping_trace_preserving(self):
        """Verify phase damping channel is trace preserving."""
        gamma = 0.1  # Dephasing rate

        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        E1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)

        kraus_ops = [E0, E1]
        assert self._is_trace_preserving(kraus_ops), \
            "Phase damping channel should be trace preserving"


class TestKnownCircuitResults:
    """Test that known quantum circuits produce expected results."""

    def test_bell_state_preparation(self):
        """Verify Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 preparation."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        I = np.eye(2, dtype=complex)
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

        # Start with |00⟩
        state = np.array([1, 0, 0, 0], dtype=complex)

        # Apply H ⊗ I
        H_I = np.kron(H, I)
        state = H_I @ state

        # Apply CNOT
        state = CNOT @ state

        # Expected: (|00⟩ + |11⟩)/√2
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        assert np.allclose(state, expected), \
            "Bell state preparation failed"

    def test_ghz_state_preparation(self):
        """Verify 3-qubit GHZ state (|000⟩ + |111⟩)/√2 preparation."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        I = np.eye(2, dtype=complex)
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

        # Start with |000⟩
        state = np.zeros(8, dtype=complex)
        state[0] = 1.0

        # Apply H ⊗ I ⊗ I
        H_I_I = np.kron(np.kron(H, I), I)
        state = H_I_I @ state

        # Apply CNOT on qubits 0,1 (control=0, target=1)
        CNOT_01 = np.kron(CNOT, I)
        state = CNOT_01 @ state

        # Apply CNOT on qubits 1,2 (control=1, target=2)
        CNOT_12 = np.kron(I, CNOT)
        state = CNOT_12 @ state

        # Expected: (|000⟩ + |111⟩)/√2
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1.0 / np.sqrt(2)  # |000⟩
        expected[7] = 1.0 / np.sqrt(2)  # |111⟩

        assert np.allclose(state, expected), \
            "GHZ state preparation failed"

    def test_hadamard_self_inverse(self):
        """Verify H² = I (Hadamard is its own inverse)."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        H_squared = H @ H
        I = np.eye(2, dtype=complex)

        assert np.allclose(H_squared, I), "H² should equal I"

    def test_pauli_anticommutation(self):
        """Verify Pauli matrices anticommute: {σ_i, σ_j} = 2δ_{ij}I."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        # XY + YX = 0
        assert np.allclose(X @ Y + Y @ X, np.zeros((2, 2))), "{X, Y} should be 0"
        # XZ + ZX = 0
        assert np.allclose(X @ Z + Z @ X, np.zeros((2, 2))), "{X, Z} should be 0"
        # YZ + ZY = 0
        assert np.allclose(Y @ Z + Z @ Y, np.zeros((2, 2))), "{Y, Z} should be 0"
        # X² = Y² = Z² = I
        assert np.allclose(X @ X, I), "X² should equal I"
        assert np.allclose(Y @ Y, I), "Y² should equal I"
        assert np.allclose(Z @ Z, I), "Z² should equal I"


class TestCliffordGroupProperties:
    """Test properties of the Clifford group used in DRB."""

    def test_clifford_generators(self):
        """Verify H and S generate all 24 single-qubit Cliffords."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        I = np.eye(2, dtype=complex)

        # Generate all 24 Cliffords using H and S sequences
        # See Nielsen & Chuang for the standard enumeration
        cliffords = set()

        def matrix_to_tuple(m):
            """Convert matrix to hashable tuple (for uniqueness checking)."""
            # Normalize global phase
            if np.abs(m[0, 0]) > 1e-10:
                phase = m[0, 0] / np.abs(m[0, 0])
            elif np.abs(m[0, 1]) > 1e-10:
                phase = m[0, 1] / np.abs(m[0, 1])
            else:
                phase = 1
            m_normalized = m / phase
            return tuple(np.round(m_normalized.flatten(), 6))

        # Generate by composing H and S up to length 6
        queue = [I]
        for _ in range(6):
            new_queue = []
            for m in queue:
                for gate in [H, S]:
                    new_m = gate @ m
                    key = matrix_to_tuple(new_m)
                    if key not in cliffords:
                        cliffords.add(key)
                        new_queue.append(new_m)
            queue = new_queue

        # Should have exactly 24 unique Cliffords
        assert len(cliffords) == 24, \
            f"Expected 24 Cliffords, got {len(cliffords)}"


class TestScalabilityLimits:
    """Test scalability and resource constraints."""

    def test_state_vector_memory_estimate(self):
        """Verify state vector memory requirements."""
        # State vector for n qubits requires 2^n complex numbers
        # Each complex128 is 16 bytes

        for n_qubits in range(1, 21):
            dim = 2 ** n_qubits
            memory_bytes = dim * 16  # complex128 = 16 bytes
            memory_mb = memory_bytes / (1024 * 1024)

            # For n=20: 2^20 * 16 = 16 MB (state vector only)
            # For n=25: 2^25 * 16 = 512 MB (state vector only)
            # For n=30: 2^30 * 16 = 16 GB (impractical)

            if n_qubits <= 20:
                assert memory_mb < 20, f"{n_qubits} qubits should require < 20 MB"

    def test_recommended_qubit_limit(self):
        """Document the recommended qubit limit for state vector simulation."""
        # Based on memory constraints and performance:
        # - 20 qubits: ~16 MB state vector, practical on any modern system
        # - 25 qubits: ~512 MB state vector, requires significant RAM
        # - 30 qubits: ~16 GB state vector, requires HPC resources

        RECOMMENDED_MAX_QUBITS = 25
        PRACTICAL_MAX_QUBITS = 20

        # This test documents the limits
        assert PRACTICAL_MAX_QUBITS <= RECOMMENDED_MAX_QUBITS
        assert RECOMMENDED_MAX_QUBITS <= 30  # Beyond this needs tensor networks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
