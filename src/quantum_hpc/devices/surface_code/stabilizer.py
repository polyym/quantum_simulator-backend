# src/quantum_hpc/devices/surface_code/stabilizer.py

"""
Surface Code Stabilizer Measurement Module

This module implements stabilizer measurements for surface code quantum error correction.
Key physics concepts:
- X stabilizers detect Z (bit-flip) errors
- Z stabilizers detect X (phase-flip) errors
- Detection events are computed as XOR of consecutive syndrome measurements
- Syndrome history is essential for proper error tracking across QEC cycles

References:
- Fowler et al., "Surface codes: Towards practical large-scale quantum computation"
- Google Quantum AI, "Suppressing quantum errors by scaling a surface code logical qubit"
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Union, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class StabilizerMeasurementResult:
    """
    Container for measured stabilizer results in a surface code cycle.

    Attributes:
        X_stabilizers: 2D array of X-stabilizer measurements (0 or 1).
                       Value of 1 indicates an error was detected.
        Z_stabilizers: 2D array of Z-stabilizer measurements (0 or 1).
        detection_events: Changes in stabilizer outcomes between this cycle and previous.
                         This is the actual syndrome used for decoding (XOR of consecutive measurements).
        measurement_fidelity: Approximate fidelity or confidence in these stabilizer measurements.
        cycle_index: Which QEC cycle this measurement belongs to.
        metadata: Additional information (timestamps, error rates, etc.)

    Physics Note:
        Detection events (not raw stabilizer values) are what the decoder uses.
        A detection event occurs when a stabilizer changes value between rounds,
        indicating an error occurred. This is computed as:
            detection_event[i] = stabilizer[cycle][i] XOR stabilizer[cycle-1][i]
    """
    X_stabilizers: List[List[int]]
    Z_stabilizers: List[List[int]]
    detection_events: Optional[Dict[str, List[List[int]]]] = None
    measurement_fidelity: float = 1.0
    cycle_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyndromeHistory:
    """
    Maintains syndrome measurement history across multiple QEC cycles.

    This is essential for proper surface code operation because:
    1. Detection events require comparing consecutive measurements
    2. The decoder needs the full syndrome history for temporal matching
    3. Measurement errors can create "vertical" error chains in spacetime

    Attributes:
        X_history: List of X-stabilizer measurements per cycle
        Z_history: List of Z-stabilizer measurements per cycle
        detection_history: List of detection events per cycle
        max_history: Maximum number of cycles to retain (for memory management)
    """
    X_history: List[List[List[int]]] = field(default_factory=list)
    Z_history: List[List[List[int]]] = field(default_factory=list)
    detection_history: List[Dict[str, List[List[int]]]] = field(default_factory=list)
    max_history: int = 100

    def add_measurement(self,
                        X_stabilizers: List[List[int]],
                        Z_stabilizers: List[List[int]]) -> Dict[str, List[List[int]]]:
        """
        Add a new measurement and compute detection events.

        Args:
            X_stabilizers: Current X-stabilizer measurements
            Z_stabilizers: Current Z-stabilizer measurements

        Returns:
            Detection events for this cycle (XOR with previous cycle)
        """
        # Compute detection events (XOR with previous measurement)
        if self.X_history and self.Z_history:
            prev_X = self.X_history[-1]
            prev_Z = self.Z_history[-1]
            X_detection = self._compute_xor(X_stabilizers, prev_X)
            Z_detection = self._compute_xor(Z_stabilizers, prev_Z)
        else:
            # First cycle: detection events are just the raw measurements
            # (compared against implicit "all zeros" initial state)
            X_detection = deepcopy(X_stabilizers)
            Z_detection = deepcopy(Z_stabilizers)

        detection_events = {
            'X': X_detection,
            'Z': Z_detection
        }

        # Store in history
        self.X_history.append(deepcopy(X_stabilizers))
        self.Z_history.append(deepcopy(Z_stabilizers))
        self.detection_history.append(detection_events)

        # Trim history if too long
        if len(self.X_history) > self.max_history:
            self.X_history.pop(0)
            self.Z_history.pop(0)
            self.detection_history.pop(0)

        return detection_events

    def _compute_xor(self,
                     current: List[List[int]],
                     previous: List[List[int]]) -> List[List[int]]:
        """Compute element-wise XOR of two 2D arrays."""
        result = []
        for i, row in enumerate(current):
            result_row = []
            for j, val in enumerate(row):
                prev_val = previous[i][j] if i < len(previous) and j < len(previous[i]) else 0
                result_row.append(val ^ prev_val)
            result.append(result_row)
        return result

    def get_detection_count(self) -> int:
        """Return total number of detection events in history."""
        count = 0
        for events in self.detection_history:
            for stab_type in ['X', 'Z']:
                if stab_type in events:
                    for row in events[stab_type]:
                        count += sum(row)
        return count

    def clear(self) -> None:
        """Clear all history."""
        self.X_history.clear()
        self.Z_history.clear()
        self.detection_history.clear()


class SurfaceCodeStabilizer:
    """
    Handle stabilizer measurement logic for a distance-d surface code.

    This class implements the stabilizer measurement protocol for surface codes,
    including proper syndrome history tracking for detection event computation.

    Physics Background:
    - A distance-d surface code has d² data qubits arranged in a 2D grid
    - X-type stabilizers are products of X operators on plaquettes (detect Z errors)
    - Z-type stabilizers are products of Z operators on plaquettes (detect X errors)
    - Ancilla qubits are used to measure stabilizers non-destructively
    - Detection events (syndrome changes) are used for decoding, not raw measurements

    Measurement Protocol (per stabilizer):
    1. Prepare ancilla in appropriate basis (|+⟩ for X-type, |0⟩ for Z-type)
    2. Entangle ancilla with data qubits via CNOT gates
    3. Measure ancilla to extract parity information
    4. Reset ancilla for next cycle
    """

    def __init__(self,
                 processor: Any,
                 distance: int,
                 ancilla_offset: Optional[int] = None,
                 track_history: bool = True):
        """
        Initialize the stabilizer measurement handler.

        Args:
            processor: A quantum processor object implementing apply_gate, measure, etc.
            distance: The code distance d (d x d data qubit grid).
            ancilla_offset: Index offset for ancilla qubits (default: d²).
            track_history: Whether to maintain syndrome history for detection events.
        """
        self.processor = processor
        self.distance = distance
        self.ancilla_offset = ancilla_offset
        self.track_history = track_history
        self._cycle_count = 0

        # If offset is not provided, assume ancillas follow data qubits
        if self.ancilla_offset is None:
            self.ancilla_offset = distance * distance

        # Calculate number of ancillas needed
        self.num_ancillas = self._calculate_num_ancillas(distance)

        # Initialize syndrome history tracker
        self.syndrome_history = SyndromeHistory() if track_history else None

        logger.debug(
            f"SurfaceCodeStabilizer initialized: distance={distance}, "
            f"ancilla_offset={self.ancilla_offset}, num_ancillas={self.num_ancillas}, "
            f"history_tracking={'enabled' if track_history else 'disabled'}"
        )

    def measure_all_stabilizers(self,
                                cycle_index: Optional[int] = None) -> StabilizerMeasurementResult:
        """
        Measure all X- and Z-type stabilizers in the surface code.

        This method performs a complete round of stabilizer measurements and
        computes detection events by comparing with the previous round.

        Args:
            cycle_index: Optional label/index for this measurement cycle.
                        If None, uses internal counter.

        Returns:
            StabilizerMeasurementResult containing:
            - Raw X and Z stabilizer outcomes
            - Detection events (XOR with previous cycle)
            - Measurement fidelity estimate
            - Cycle metadata

        Physics Note:
            The decoder should use detection_events, not raw stabilizer values.
            Detection events indicate where errors occurred between rounds.
        """
        try:
            # Use internal counter if cycle_index not provided
            if cycle_index is None:
                cycle_index = self._cycle_count
            self._cycle_count += 1

            # Prepare lists to store measurement outcomes
            x_stabilizers = [[0] * (self.distance - 1) for _ in range(self.distance - 1)]
            z_stabilizers = [[0] * (self.distance - 1) for _ in range(self.distance - 1)]

            # Measure X-type stabilizers (detect Z errors)
            for i in range(self.distance - 1):
                for j in range(self.distance - 1):
                    x_stabilizers[i][j] = self._measure_x_stabilizer(i, j)

            # Measure Z-type stabilizers (detect X errors)
            for i in range(self.distance - 1):
                for j in range(self.distance - 1):
                    z_stabilizers[i][j] = self._measure_z_stabilizer(i, j)

            # Compute detection events using syndrome history
            detection_events = None
            if self.track_history and self.syndrome_history is not None:
                detection_events = self.syndrome_history.add_measurement(
                    x_stabilizers, z_stabilizers
                )

            # Construct metadata
            metadata = {
                "cycle_index": cycle_index,
                "distance": self.distance,
                "timestamp": self._get_timestamp(),
                "total_X_detections": sum(sum(row) for row in (detection_events.get('X', []) if detection_events else x_stabilizers)),
                "total_Z_detections": sum(sum(row) for row in (detection_events.get('Z', []) if detection_events else z_stabilizers)),
                "history_length": len(self.syndrome_history.X_history) if self.syndrome_history else 0
            }

            return StabilizerMeasurementResult(
                X_stabilizers=x_stabilizers,
                Z_stabilizers=z_stabilizers,
                detection_events=detection_events,
                measurement_fidelity=1.0,  # Could be calculated from error model
                cycle_index=cycle_index,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error measuring all stabilizers at cycle {cycle_index}: {e}")
            raise

    def reset_history(self) -> None:
        """Reset syndrome history for a new experiment."""
        if self.syndrome_history:
            self.syndrome_history.clear()
        self._cycle_count = 0
        logger.debug("Syndrome history reset")

    def _measure_x_stabilizer(self, row: int, col: int) -> int:
        """
        Measure a single X-type stabilizer at grid position (row, col).

        An X stabilizer typically involves a set of four data qubits arranged
        in a plaquette, plus one ancilla qubit prepared in |+>.

        Returns:
            The measured outcome (0 or 1).
        """
        try:
            # Identify the data qubits involved in this X-type stabilizer
            data_qubits = self._get_x_stabilizer_qubits(row, col)

            # Identify the ancilla qubit
            ancilla_qubit = self._get_ancilla_index(row, col, is_x=True)

            # 1) Prepare ancilla in the |+> state
            #    This might require calling processor.apply_gate('H', [ancilla_qubit]) or
            #    a specialized method. We'll assume it's done implicitly or as needed.

            # 2) Apply CNOT(ancilla -> data) for X stabilizer measurement
            for dq in data_qubits:
                # If measuring X stabilizer, ancilla is the control in X-basis measurement
                # Some surface code definitions invert this; adapt as needed.
                self.processor.apply_gate('CNOT', [ancilla_qubit, dq])

            # 3) Measure ancilla in X basis
            results, fidelity = self.processor.measure([ancilla_qubit], basis="X")
            return results[0] if results else 0
        except Exception as e:
            logger.error(f"Error measuring X stabilizer at ({row}, {col}): {e}")
            return 0

    def _measure_z_stabilizer(self, row: int, col: int) -> int:
        """
        Measure a single Z-type stabilizer at grid position (row, col).

        A Z stabilizer typically involves four data qubits in a plaquette,
        plus one ancilla qubit prepared in |0>.

        Returns:
            The measured outcome (0 or 1).
        """
        try:
            # Identify data qubits
            data_qubits = self._get_z_stabilizer_qubits(row, col)

            # Identify ancilla qubit
            ancilla_qubit = self._get_ancilla_index(row, col, is_x=False)

            # 1) Prepare ancilla in |0> state
            #    Could be an explicit 'reset' + measure or a known initial condition.

            # 2) Apply CNOT(data -> ancilla) for Z stabilizer measurement
            for dq in data_qubits:
                # If measuring Z stabilizer, data qubit is typically the control
                self.processor.apply_gate('CNOT', [dq, ancilla_qubit])

            # 3) Measure ancilla in Z basis
            results, fidelity = self.processor.measure([ancilla_qubit], basis="Z")
            return results[0] if results else 0
        except Exception as e:
            logger.error(f"Error measuring Z stabilizer at ({row}, {col}): {e}")
            return 0

    def _get_x_stabilizer_qubits(self, row: int, col: int) -> List[int]:
        """
        Identify which data qubits are part of the X-type stabilizer at (row, col).

        For a typical surface code, the data qubits in an X plaquette are:
          (row, col), (row, col+1), (row+1, col), (row+1, col+1),
        mapped into a linear index according to your layout.

        Returns:
            A list of data qubit indices.
        """
        try:
            top_left = self._map_data_qubit(row, col)
            top_right = self._map_data_qubit(row, col + 1)
            bot_left = self._map_data_qubit(row + 1, col)
            bot_right = self._map_data_qubit(row + 1, col + 1)
            return [top_left, top_right, bot_left, bot_right]
        except Exception as e:
            logger.error(f"Error determining X-stabilizer qubits at ({row},{col}): {e}")
            return []

    def _get_z_stabilizer_qubits(self, row: int, col: int) -> List[int]:
        """
        Identify which data qubits are part of the Z-type stabilizer at (row, col).
        Typically the same pattern as X-type, but we keep a separate method
        in case your mapping or connectivity differs.
        """
        try:
            top_left = self._map_data_qubit(row, col)
            top_right = self._map_data_qubit(row, col + 1)
            bot_left = self._map_data_qubit(row + 1, col)
            bot_right = self._map_data_qubit(row + 1, col + 1)
            return [top_left, top_right, bot_left, bot_right]
        except Exception as e:
            logger.error(f"Error determining Z-stabilizer qubits at ({row},{col}): {e}")
            return []

    def _map_data_qubit(self, row: int, col: int) -> int:
        """
        Convert 2D coordinates (row, col) in a d x d grid to a linear
        index for your processor's data qubits.
        """
        return row * self.distance + col

    def _get_ancilla_index(self, row: int, col: int, is_x: bool) -> int:
        """
        Compute which ancilla qubit is responsible for the stabilizer at (row, col).

        This is a simple scheme that assigns one ancilla per plaquette.
        For an X-stabilizer, we might map to a different offset than Z-stabilizer,
        or keep it uniform.

        is_x: Flag to distinguish if it's an X-type or Z-type ancilla (if needed).
        """
        # Simplest approach: place ancillas for each plaquette in row-major order
        # e.g., ancilla_offset + row*(distance-1) + col
        # If you have separate X vs. Z ancillas, adapt accordingly.
        ancilla_base = 0
        # If you separate X and Z ancillas, you might do:
        # ancilla_base = 0 if is_x else (distance-1)*(distance-1) for a code with separate sets.

        # For a single unified ancilla block, just do:
        index_in_plaquette_grid = row * (self.distance - 1) + col
        return self.ancilla_offset + ancilla_base + index_in_plaquette_grid

    def _get_timestamp(self) -> float:
        """Helper to generate a timestamp (could also come from the processor or HPC environment)."""
        import time
        return time.time()

    def _calculate_num_ancillas(self, distance: int) -> int:
        """
        Optional helper method to estimate how many ancillas are needed.
        Typically, a distance-d surface code has (d-1)*d + d*(d-1) = 2*d*(d-1) plaquettes,
        but some layouts separate X/Z ancillas. Adjust as needed.
        """
        # Simple approach: each of the (distance-1)*(distance-1) squares has one ancilla,
        # and maybe you need a second set for the other type. For demonstration, we assume 2 sets.
        # If you unify them, reduce accordingly.
        # Example: separate X and Z ancillas => 2*(distance-1)*(distance-1).
        # If using a single ancilla for both, use just (distance-1)*(distance-1).
        return 2 * (distance - 1) * (distance - 1)
