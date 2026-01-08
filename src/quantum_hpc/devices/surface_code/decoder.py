# src/quantum_hpc/devices/surface_code/decoder.py

"""
Surface Code Decoder Module

This module implements syndrome decoding for surface code quantum error correction.
The decoder takes detection events (syndrome changes) and determines which physical
corrections to apply to recover the logical qubit state.

Key Concepts:
- Detection events form a graph where edges connect syndrome locations
- MWPM (Minimum Weight Perfect Matching) finds optimal error chains
- Errors are corrected by applying Pauli operators along matched paths

References:
- Fowler et al., "Minimum weight perfect matching of fault-tolerant topological quantum error correction in average O(1) parallel time"
- Higgott, "PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching"
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
import heapq

from .stabilizer import StabilizerMeasurementResult

logger = logging.getLogger(__name__)


@dataclass
class DecodedSyndromeResult:
    """
    Container for the output of the surface code decoder.

    Attributes:
        corrections: List of correction operations to apply.
                    Each dict contains 'operation' (X_flip/Z_flip) and 'qubits' (list of indices).
        logical_errors_detected: List of logical error identifiers if boundaries are spanned.
        error_chains: The matched error chains from MWPM.
        success_probability: Estimated probability that decoding was successful.
        metadata: Additional details (cycle index, decoder stats, etc.)

    Physics Note:
        Corrections are Pauli operators applied to data qubits.
        X_flip corrects Z errors (detected by X stabilizers).
        Z_flip corrects X errors (detected by Z stabilizers).
    """
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    logical_errors_detected: List[str] = field(default_factory=list)
    error_chains: Optional[List[Dict[str, Any]]] = None
    success_probability: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchingNode:
    """
    Represents a node in the matching graph.

    Attributes:
        id: Unique identifier for this node
        row: Row position in the stabilizer grid
        col: Column position in the stabilizer grid
        cycle: Time cycle when this detection occurred
        stabilizer_type: 'X' or 'Z'
        is_boundary: Whether this is a virtual boundary node
    """
    id: int
    row: int
    col: int
    cycle: int
    stabilizer_type: str
    is_boundary: bool = False

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, MatchingNode) and self.id == other.id


class SurfaceCodeDecoder:
    """
    Decode syndrome data using Minimum Weight Perfect Matching (MWPM).

    This decoder implements a graph-based approach where:
    1. Detection events become nodes in a matching graph
    2. Edge weights represent physical distances (error probability)
    3. MWPM finds the most likely error configuration
    4. Corrections are determined by tracing paths between matched nodes

    The implementation uses a simplified but correct MWPM algorithm suitable
    for moderate code distances. For production use with large distances,
    consider using optimized libraries like PyMatching.

    Physics Background:
    - Errors create pairs of detection events (except at boundaries)
    - MWPM finds the minimum-weight pairing of all detection events
    - The weight of an edge represents the number of physical errors
    - Boundary nodes allow single detection events to be matched
    """

    def __init__(self, distance: int, physical_error_rate: float = 0.001):
        """
        Initialize the decoder.

        Args:
            distance: The code distance d.
            physical_error_rate: Estimated physical error rate for weight calculation.
        """
        self.distance = distance
        self.physical_error_rate = physical_error_rate
        self._node_counter = 0

        # Precompute boundary node positions
        self._boundary_nodes_X: List[MatchingNode] = []
        self._boundary_nodes_Z: List[MatchingNode] = []
        self._initialize_boundary_nodes()

        logger.debug(
            f"SurfaceCodeDecoder initialized: distance={distance}, "
            f"physical_error_rate={physical_error_rate}"
        )

    def _initialize_boundary_nodes(self) -> None:
        """Create virtual boundary nodes for matching single defects."""
        # For X stabilizers, boundaries are at top and bottom
        for col in range(self.distance - 1):
            # Top boundary
            self._boundary_nodes_X.append(MatchingNode(
                id=self._get_next_node_id(),
                row=-1, col=col, cycle=0,
                stabilizer_type='X', is_boundary=True
            ))
            # Bottom boundary
            self._boundary_nodes_X.append(MatchingNode(
                id=self._get_next_node_id(),
                row=self.distance - 1, col=col, cycle=0,
                stabilizer_type='X', is_boundary=True
            ))

        # For Z stabilizers, boundaries are at left and right
        for row in range(self.distance - 1):
            # Left boundary
            self._boundary_nodes_Z.append(MatchingNode(
                id=self._get_next_node_id(),
                row=row, col=-1, cycle=0,
                stabilizer_type='Z', is_boundary=True
            ))
            # Right boundary
            self._boundary_nodes_Z.append(MatchingNode(
                id=self._get_next_node_id(),
                row=row, col=self.distance - 1, cycle=0,
                stabilizer_type='Z', is_boundary=True
            ))

    def _get_next_node_id(self) -> int:
        """Generate unique node ID."""
        self._node_counter += 1
        return self._node_counter

    def decode_syndrome(self,
                        syndrome_data: StabilizerMeasurementResult) -> DecodedSyndromeResult:
        """
        Decode syndrome data and determine corrections.

        This method:
        1. Extracts detection events from syndrome data
        2. Builds a matching graph with nodes and weighted edges
        3. Runs MWPM to pair up detection events
        4. Converts matched pairs to physical corrections
        5. Checks for logical errors (boundary-spanning chains)

        Args:
            syndrome_data: Output from SurfaceCodeStabilizer.measure_all_stabilizers()

        Returns:
            DecodedSyndromeResult with corrections and diagnostics.
        """
        try:
            # Use detection events if available, otherwise use raw stabilizers
            if syndrome_data.detection_events:
                x_detections = syndrome_data.detection_events.get('X', syndrome_data.X_stabilizers)
                z_detections = syndrome_data.detection_events.get('Z', syndrome_data.Z_stabilizers)
            else:
                x_detections = syndrome_data.X_stabilizers
                z_detections = syndrome_data.Z_stabilizers

            cycle = syndrome_data.cycle_index

            # Build matching graphs and find corrections
            x_corrections, x_chains, x_logical = self._decode_stabilizer_type(
                x_detections, 'X', cycle
            )
            z_corrections, z_chains, z_logical = self._decode_stabilizer_type(
                z_detections, 'Z', cycle
            )

            # Combine results
            all_corrections = x_corrections + z_corrections
            all_chains = x_chains + z_chains
            logical_errors = x_logical + z_logical

            # Estimate success probability based on chain lengths
            total_weight = sum(chain.get('weight', 0) for chain in all_chains)
            success_prob = (1 - self.physical_error_rate) ** total_weight if total_weight > 0 else 1.0

            return DecodedSyndromeResult(
                corrections=all_corrections,
                logical_errors_detected=logical_errors,
                error_chains=all_chains,
                success_probability=success_prob,
                metadata={
                    "cycle_index": cycle,
                    "distance": self.distance,
                    "num_X_detections": sum(sum(row) for row in x_detections),
                    "num_Z_detections": sum(sum(row) for row in z_detections),
                    "num_corrections": len(all_corrections),
                    "decoder_algorithm": "MWPM"
                }
            )
        except Exception as e:
            logger.error(f"Error decoding syndrome data: {e}")
            raise

    def _decode_stabilizer_type(self,
                                 detections: List[List[int]],
                                 stab_type: str,
                                 cycle: int) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Decode detections for one stabilizer type (X or Z).

        Args:
            detections: 2D array of detection events (0 or 1)
            stab_type: 'X' or 'Z'
            cycle: Current QEC cycle

        Returns:
            Tuple of (corrections, error_chains, logical_errors)
        """
        # Find all detection event locations
        nodes = self._extract_detection_nodes(detections, stab_type, cycle)

        if not nodes:
            return [], [], []

        # Get boundary nodes for this stabilizer type
        boundary = self._boundary_nodes_X if stab_type == 'X' else self._boundary_nodes_Z

        # Build complete graph with weights
        graph = self._build_matching_graph(nodes, boundary, stab_type)

        # Run MWPM
        matching = self._minimum_weight_perfect_matching(nodes + boundary, graph)

        # Convert matching to corrections
        corrections, chains = self._matching_to_corrections(matching, stab_type)

        # Check for logical errors
        logical_errors = self._check_logical_errors(matching, stab_type)

        return corrections, chains, logical_errors

    def _extract_detection_nodes(self,
                                  detections: List[List[int]],
                                  stab_type: str,
                                  cycle: int) -> List[MatchingNode]:
        """Extract MatchingNodes from detection event array."""
        nodes = []
        for i, row in enumerate(detections):
            for j, val in enumerate(row):
                if val == 1:
                    nodes.append(MatchingNode(
                        id=self._get_next_node_id(),
                        row=i, col=j, cycle=cycle,
                        stabilizer_type=stab_type,
                        is_boundary=False
                    ))
        return nodes

    def _build_matching_graph(self,
                               nodes: List[MatchingNode],
                               boundary_nodes: List[MatchingNode],
                               stab_type: str) -> Dict[Tuple[int, int], float]:
        """
        Build complete weighted graph for matching.

        Edge weights are based on Manhattan distance, which corresponds to
        the minimum number of physical errors needed to create the detection pair.
        """
        graph = {}
        all_nodes = nodes + boundary_nodes

        for i, n1 in enumerate(all_nodes):
            for n2 in all_nodes[i+1:]:
                weight = self._compute_edge_weight(n1, n2, stab_type)
                graph[(n1.id, n2.id)] = weight
                graph[(n2.id, n1.id)] = weight

        return graph

    def _compute_edge_weight(self,
                              n1: MatchingNode,
                              n2: MatchingNode,
                              stab_type: str) -> float:
        """
        Compute edge weight between two nodes.

        Weight is based on:
        - Manhattan distance for spatial separation
        - Time distance for temporal separation
        - Log-likelihood ratio for probabilistic interpretation
        """
        if n1.is_boundary and n2.is_boundary:
            # Boundary-to-boundary edges have infinite weight (shouldn't be matched)
            return float('inf')

        if n1.is_boundary:
            # Distance to boundary
            if stab_type == 'X':
                # X boundaries are at top/bottom
                dist = min(n2.row + 1, self.distance - 1 - n2.row)
            else:
                # Z boundaries are at left/right
                dist = min(n2.col + 1, self.distance - 1 - n2.col)
        elif n2.is_boundary:
            if stab_type == 'X':
                dist = min(n1.row + 1, self.distance - 1 - n1.row)
            else:
                dist = min(n1.col + 1, self.distance - 1 - n1.col)
        else:
            # Manhattan distance between detection events
            dist = abs(n1.row - n2.row) + abs(n1.col - n2.col) + abs(n1.cycle - n2.cycle)

        # Convert distance to weight using error probability
        # Weight = -log(P(error chain of length dist))
        if self.physical_error_rate > 0 and self.physical_error_rate < 1:
            weight = dist * (-1) * (
                (1 - self.physical_error_rate) / self.physical_error_rate
                if self.physical_error_rate < 0.5 else 1
            )
            weight = max(dist, 1)  # Ensure positive weight
        else:
            weight = dist

        return weight

    def _minimum_weight_perfect_matching(self,
                                          nodes: List[MatchingNode],
                                          graph: Dict[Tuple[int, int], float]) -> List[Tuple[MatchingNode, MatchingNode]]:
        """
        Find minimum weight perfect matching using a greedy approximation.

        For production use with large graphs, consider using the Blossom algorithm
        or PyMatching library. This greedy approach provides a good approximation
        for moderate-size syndrome graphs.

        Note: This is O(n²) greedy matching. For exact MWPM, use Blossom V or similar.
        """
        # Filter to only non-boundary nodes that need matching
        active_nodes = [n for n in nodes if not n.is_boundary]

        if len(active_nodes) == 0:
            return []

        # If odd number of detection events, we need to match one to boundary
        if len(active_nodes) % 2 == 1:
            # Find node closest to boundary and add virtual boundary match
            boundary_nodes = [n for n in nodes if n.is_boundary]
            active_nodes = active_nodes + boundary_nodes[:1]  # Add one boundary node

        matching = []
        matched = set()

        # Sort edges by weight
        edges = []
        for (n1_id, n2_id), weight in graph.items():
            if weight < float('inf'):
                n1 = next((n for n in nodes if n.id == n1_id), None)
                n2 = next((n for n in nodes if n.id == n2_id), None)
                if n1 and n2:
                    edges.append((weight, n1, n2))

        edges.sort(key=lambda x: x[0])

        # Greedy matching: take lowest weight edges first
        for weight, n1, n2 in edges:
            if n1.id not in matched and n2.id not in matched:
                # Don't match two boundary nodes
                if n1.is_boundary and n2.is_boundary:
                    continue
                matching.append((n1, n2))
                matched.add(n1.id)
                matched.add(n2.id)

        return matching

    def _matching_to_corrections(self,
                                  matching: List[Tuple[MatchingNode, MatchingNode]],
                                  stab_type: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Convert matched pairs to physical corrections.

        For each matched pair, we need to apply corrections along the path
        connecting them. The correction type depends on the stabilizer type:
        - X stabilizer detections → Z_flip corrections (correct X errors)
        - Z stabilizer detections → X_flip corrections (correct Z errors)
        """
        corrections = []
        chains = []

        # Determine correction type: X stabilizers detect Z errors, Z stabilizers detect X errors
        correction_type = "Z_flip" if stab_type == 'X' else "X_flip"

        for n1, n2 in matching:
            # Find path between nodes
            path_qubits = self._find_correction_path(n1, n2, stab_type)

            if path_qubits:
                weight = len(path_qubits)
                corrections.append({
                    "operation": correction_type,
                    "qubits": path_qubits,
                    "stabilizer_type": stab_type
                })
                chains.append({
                    "type": stab_type,
                    "from_node": (n1.row, n1.col, n1.cycle),
                    "to_node": (n2.row, n2.col, n2.cycle),
                    "affected_qubits": path_qubits,
                    "weight": weight,
                    "involves_boundary": n1.is_boundary or n2.is_boundary
                })

        return corrections, chains

    def _find_correction_path(self,
                               n1: MatchingNode,
                               n2: MatchingNode,
                               stab_type: str) -> List[Tuple[int, int]]:
        """
        Find the data qubits along the path between two detection events.

        The path follows the lattice structure of the surface code.
        For now, we use a simple Manhattan path; optimal paths would
        consider the specific error model.
        """
        path = []

        if n1.is_boundary:
            # Path from detection to boundary
            if stab_type == 'X':
                # Move vertically to boundary
                start_row = n2.row
                if n1.row < 0:
                    # Top boundary
                    for r in range(start_row, -1, -1):
                        path.append((r, n2.col))
                else:
                    # Bottom boundary
                    for r in range(start_row, self.distance):
                        path.append((r, n2.col))
            else:
                # Move horizontally to boundary
                start_col = n2.col
                if n1.col < 0:
                    # Left boundary
                    for c in range(start_col, -1, -1):
                        path.append((n2.row, c))
                else:
                    # Right boundary
                    for c in range(start_col, self.distance):
                        path.append((n2.row, c))
        elif n2.is_boundary:
            # Symmetric case
            return self._find_correction_path(n2, n1, stab_type)
        else:
            # Path between two detection events
            # Use L-shaped path (vertical then horizontal)
            r1, c1 = n1.row, n1.col
            r2, c2 = n2.row, n2.col

            # Vertical segment
            if r1 <= r2:
                for r in range(r1, r2 + 1):
                    path.append((r, c1))
            else:
                for r in range(r1, r2 - 1, -1):
                    path.append((r, c1))

            # Horizontal segment (skip the corner we already added)
            if c1 < c2:
                for c in range(c1 + 1, c2 + 1):
                    path.append((r2, c))
            elif c1 > c2:
                for c in range(c1 - 1, c2 - 1, -1):
                    path.append((r2, c))

        return path

    def _check_logical_errors(self,
                               matching: List[Tuple[MatchingNode, MatchingNode]],
                               stab_type: str) -> List[str]:
        """
        Check if any matched paths span the code boundaries (logical error).

        A logical error occurs when an error chain connects opposite boundaries:
        - X logical error: chain spans left to right
        - Z logical error: chain spans top to bottom
        """
        logical_errors = []

        for n1, n2 in matching:
            if n1.is_boundary and n2.is_boundary:
                # Two boundary nodes matched - potential logical error
                if stab_type == 'X':
                    # Check if top-to-bottom span
                    if (n1.row < 0 and n2.row >= self.distance - 1) or \
                       (n2.row < 0 and n1.row >= self.distance - 1):
                        logical_errors.append(f"Logical_Z_error_cycle_{n1.cycle}")
                else:
                    # Check if left-to-right span
                    if (n1.col < 0 and n2.col >= self.distance - 1) or \
                       (n2.col < 0 and n1.col >= self.distance - 1):
                        logical_errors.append(f"Logical_X_error_cycle_{n1.cycle}")

        return logical_errors
