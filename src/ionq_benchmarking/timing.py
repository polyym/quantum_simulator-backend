# src/ionq_benchmarking/timing.py

"""
Timing utilities for IonQ-like quantum computing processes,
including gate execution times, overheads (state prep/measurement),
and application-level timing analysis.
"""

import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class GateTiming:
    """
    Stores timing information for various gate operations according to IonQ data:
      - single_qubit: Time (in microseconds) for single-qubit gates
      - zz_gate: Time (in microseconds) for a two-qubit ZZ gate (or similar)
      - state_prep: Time (in milliseconds) for state preparation
      - measurement: Time (in milliseconds) for measurement
      - aod_switch: Additional overhead (in milliseconds) for AOD switching
    """
    single_qubit: float = 110.0
    zz_gate: float = 900.0
    state_prep: float = 0.5
    measurement: float = 0.5
    aod_switch: float = 0.1

@dataclass
class CircuitTiming:
    """
    Captures execution timing results for a quantum circuit:
      - total_time: overall execution time (ms)
      - gate_time: sum of gate durations (ms)
      - overhead_time: sum of overhead durations (ms)
      - shots: number of shots or runs
    """
    total_time: float = 0.0
    gate_time: float = 0.0
    overhead_time: float = 0.0
    shots: int = 100

class TimingAnalyzer:
    """
    Analyzes and tracks timing performance of IonQ-like quantum circuits.
    Uses a simplified model of gate durations, overhead times, and optional
    AOD switching delays.
    """

    def __init__(self):
        # Default gate timing as suggested in IonQ papers
        self.gate_timing = GateTiming()
        self.circuit_timings: Dict[str, CircuitTiming] = {}
        self.start_time: Optional[datetime] = None
        self.current_circuit: Optional[str] = None

    def start_circuit(self, circuit_id: str) -> None:
        """
        Begin timing a new circuit by storing the start time and circuit identifier.

        Args:
            circuit_id: A unique identifier for the circuit being timed.
        """
        self.start_time = datetime.now()
        self.current_circuit = circuit_id
        logger.debug(f"Timing started for circuit '{circuit_id}' at {self.start_time}.")

    def end_circuit(self, shots: int = 100) -> None:
        """
        Conclude timing for the current circuit and record the total elapsed time.

        Args:
            shots: The number of shots or runs executed for this circuit.
        """
        if self.start_time and self.current_circuit:
            end_time = datetime.now()
            execution_time_ms = (end_time - self.start_time).total_seconds() * 1000.0
            self.circuit_timings[self.current_circuit] = CircuitTiming(
                total_time=execution_time_ms,
                shots=shots
            )
            logger.info(
                f"Circuit '{self.current_circuit}' ended. "
                f"Execution time: {execution_time_ms:.2f} ms for {shots} shots."
            )
            self.start_time = None
            self.current_circuit = None
        else:
            logger.warning("end_circuit called but no circuit was being timed.")

    def analyze_circuit_timing(self,
                               circuit: List[Dict],
                               include_overhead: bool = True) -> Dict[str, float]:
        """
        Estimate the execution time of a quantum circuit based on IonQ-like
        gate durations, overhead times, and potential AOD switching overhead.

        Args:
            circuit: A list of gate dictionaries, each with 'type' and 'qubits'.
            include_overhead: If True, add state prep and measurement overhead to the total.

        Returns:
            A dictionary of timing data, including total_time_ms, gate_time_ms,
            switching_time_ms, overhead_time_ms, and gate_percentage.
        """
        try:
            gate_time_ms = 0.0
            switching_time_ms = 0.0
            overhead_time_ms = 0.0

            # Track the last set of qubits used to estimate AOD switching overhead
            prev_qubits = set()

            for op in circuit:
                op_type = op.get('type', 'unknown')
                qubits = set(op.get('qubits', []))

                # Gate time
                if op_type in ['H', 'X', 'Y', 'Z', 'T', 'S']:
                    # Single-qubit gate
                    gate_time_ms += self.gate_timing.single_qubit
                elif op_type == 'ZZ' or op_type in ['CNOT', 'CZ', 'XX', 'YY']:
                    # Two-qubit or multi-qubit gate, approximate with zz_gate timing
                    gate_time_ms += self.gate_timing.zz_gate

                # AOD switching overhead if the qubits changed
                if qubits != prev_qubits:
                    switching_time_ms += self.gate_timing.aod_switch
                prev_qubits = qubits

            if include_overhead:
                overhead_time_ms = (self.gate_timing.state_prep + 
                                    self.gate_timing.measurement)

            total_time_ms = gate_time_ms + switching_time_ms + overhead_time_ms

            if total_time_ms > 0:
                gate_percentage = (gate_time_ms / total_time_ms) * 100.0
            else:
                gate_percentage = 0.0

            logger.debug(
                "Circuit timing analysis: "
                f"gate_time={gate_time_ms:.2f} ms, switching_time={switching_time_ms:.2f} ms, "
                f"overhead_time={overhead_time_ms:.2f} ms, total_time={total_time_ms:.2f} ms."
            )

            return {
                "total_time_ms": total_time_ms,
                "gate_time_ms": gate_time_ms,
                "switching_time_ms": switching_time_ms,
                "overhead_time_ms": overhead_time_ms,
                "gate_percentage": gate_percentage
            }

        except Exception as e:
            logger.error(f"Error in timing analysis: {e}")
            return {}

class ApplicationTimingTracker:
    """
    Tracks timing for IonQ-like application benchmarks, storing
    estimated or measured times for various widths/gates.
    """

    def __init__(self):
        self.analyzer = TimingAnalyzer()
        self.application_timings: Dict[str, Dict[str, float]] = {}

    def track_application(self,
                          name: str,
                          circuit_width: int,
                          num_gates: int,
                          shots: int = 100):
        """
        Record timing estimates for a particular application scenario.

        Args:
            name: Name of the application (e.g., 'phase_estimation').
            circuit_width: Number of qubits used.
            num_gates: Number of gates in the circuit.
            shots: The number of shots the circuit runs.
        """
        try:
            circuit_id = f"{name}_{circuit_width}"
            self.analyzer.start_circuit(circuit_id)

            # Rough estimate: 70% single-qubit gates, 30% two-qubit gates
            gate_time_ms = (
                num_gates * (
                    0.7 * self.analyzer.gate_timing.single_qubit +
                    0.3 * self.analyzer.gate_timing.zz_gate
                )
            )
            overhead_time_ms = (self.analyzer.gate_timing.state_prep +
                                self.analyzer.gate_timing.measurement)

            # Some fraction of gate_count * aod_switch for partial overhead
            switching_time_ms = num_gates * self.analyzer.gate_timing.aod_switch * 0.1
            total_time_ms = gate_time_ms + overhead_time_ms + switching_time_ms

            # Store into application_timings
            self.application_timings[circuit_id] = {
                "circuit_width": circuit_width,
                "num_gates": num_gates,
                "shots": shots,
                "estimated_time_ms": total_time_ms,
                "gate_time_percentage": (gate_time_ms / total_time_ms * 100.0) if total_time_ms > 0 else 0.0
            }

            # End circuit timing
            self.analyzer.end_circuit(shots)
        except Exception as e:
            logger.error(f"Error tracking application '{name}' with width={circuit_width}: {e}")

    def get_application_statistics(self) -> Dict[str, float]:
        """
        Return basic statistics (mean, max, min, std) over all recorded application times.

        Returns:
            A dictionary with stats: average_time_ms, max_time_ms, min_time_ms, std_dev_ms, total_applications.
        """
        if not self.application_timings:
            logger.warning("No application timings recorded.")
            return {}

        times = [data["estimated_time_ms"] for data in self.application_timings.values()]
        return {
            "average_time_ms": float(np.mean(times)),
            "max_time_ms": float(np.max(times)),
            "min_time_ms": float(np.min(times)),
            "std_dev_ms": float(np.std(times)),
            "total_applications": len(times)
        }

    def get_timing_by_width(self) -> Dict[int, Dict[str, float]]:
        """
        Group timings by circuit width, calculating average and std dev for each width.

        Returns:
            A dict mapping circuit_width -> {"average_time_ms", "std_dev_ms", "num_circuits"}.
        """
        width_groups: Dict[int, List[float]] = {}

        for circuit_id, data in self.application_timings.items():
            width = data["circuit_width"]
            if width not in width_groups:
                width_groups[width] = []
            width_groups[width].append(data["estimated_time_ms"])

        results = {}
        for w, arr in width_groups.items():
            arr_np = np.array(arr, dtype=float)
            results[w] = {
                "average_time_ms": float(np.mean(arr_np)),
                "std_dev_ms": float(np.std(arr_np)),
                "num_circuits": len(arr_np)
            }
        return results
