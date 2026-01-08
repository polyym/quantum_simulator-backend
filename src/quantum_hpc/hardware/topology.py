# src/quantum_hpc/hardware/topology.py

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

@dataclass
class QubitPosition:
    """
    Represents the physical coordinates of a qubit within the topology.

    Attributes:
        qubit_id: Unique identifier for the qubit (e.g., an integer index).
        x: X-coordinate (e.g., row index or physical X-axis location).
        y: Y-coordinate (e.g., column index or physical Y-axis location).
        z: Optional Z-coordinate (for 3D layouts).
    """
    qubit_id: int
    x: float
    y: float
    z: Optional[float] = None

@dataclass
class ConnectivityLink:
    """
    Represents a connection between two qubits, along with any associated data
    (like coupling strength, expected gate errors, distance, etc.).

    Attributes:
        qubit_a: Identifier for the first qubit.
        qubit_b: Identifier for the second qubit.
        distance: Physical distance between the qubits (for noise models).
        coupling_strength: Optional measure of how strongly they're coupled.
        error_rate: Optional link-specific error rate (e.g., crosstalk).
        metadata: Arbitrary extra data about this link.
    """
    qubit_a: int
    qubit_b: int
    distance: float = 1.0
    coupling_strength: Optional[float] = None
    error_rate: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)

class QuantumTopology:
    """
    Manages physical qubit layout and connectivity for a quantum processor or simulator.

    Typical usage might include:
      - Lattice-based layouts (2D grid for superconducting qubits).
      - Full or partial connectivity (ion trap, photonic, etc.).
      - Searching for neighbors or computing distances (used by HPC or noise models).
    """

    def __init__(self):
        self.qubit_positions: Dict[int, QubitPosition] = {}
        self.connectivity_map: Dict[int, List[int]] = {}
        self.links: List[ConnectivityLink] = []

    def add_qubit_position(self, qubit_id: int, x: float, y: float, z: Optional[float] = None):
        """
        Add or update the position of a qubit in the topology.

        Args:
            qubit_id: Unique identifier for the qubit.
            x, y, z: Coordinates. z is optional for 3D topologies.
        """
        self.qubit_positions[qubit_id] = QubitPosition(qubit_id, x, y, z)
        if qubit_id not in self.connectivity_map:
            self.connectivity_map[qubit_id] = []
        logger.debug(f"Added/updated qubit position: {self.qubit_positions[qubit_id]}")

    def add_connectivity(self,
                         qubit_a: int,
                         qubit_b: int,
                         distance: float = 1.0,
                         coupling_strength: Optional[float] = None,
                         error_rate: Optional[float] = None,
                         metadata: Optional[Dict[str, float]] = None):
        """
        Define a bidirectional link between two qubits.

        Args:
            qubit_a: First qubit ID.
            qubit_b: Second qubit ID.
            distance: Physical or logical distance between them.
            coupling_strength: Coupling or interaction strength (for gating, crosstalk).
            error_rate: Link-specific error rate (optional).
            metadata: Arbitrary extra link data.
        """
        if qubit_a not in self.qubit_positions or qubit_b not in self.qubit_positions:
            logger.warning(
                f"Cannot add connectivity: qubit {qubit_a} or {qubit_b} not found."
            )
            return
        
        # Add to adjacency list
        if qubit_b not in self.connectivity_map[qubit_a]:
            self.connectivity_map[qubit_a].append(qubit_b)
        if qubit_a not in self.connectivity_map[qubit_b]:
            self.connectivity_map[qubit_b].append(qubit_a)

        # Record link data
        link_meta = metadata if metadata else {}
        link = ConnectivityLink(
            qubit_a=qubit_a,
            qubit_b=qubit_b,
            distance=distance,
            coupling_strength=coupling_strength,
            error_rate=error_rate,
            metadata=link_meta
        )
        self.links.append(link)

        logger.debug(
            f"Added connectivity: Q{qubit_a} <-> Q{qubit_b}, distance={distance}, "
            f"coupling_strength={coupling_strength}, error_rate={error_rate}"
        )

    def get_neighbors(self, qubit_id: int) -> List[int]:
        """
        Return a list of qubits directly connected to the given qubit.
        """
        return self.connectivity_map.get(qubit_id, [])

    def get_position(self, qubit_id: int) -> Optional[QubitPosition]:
        """
        Retrieve the position object for a given qubit, or None if not found.
        """
        return self.qubit_positions.get(qubit_id)

    def get_link_data(self, qubit_a: int, qubit_b: int) -> Optional[ConnectivityLink]:
        """
        Retrieve link info (distance, coupling_strength, etc.) for the connection between qubit_a and qubit_b.
        Returns None if no direct connection exists.
        """
        for link in self.links:
            if (link.qubit_a == qubit_a and link.qubit_b == qubit_b) or \
               (link.qubit_a == qubit_b and link.qubit_b == qubit_a):
                return link
        return None

    def build_2d_lattice(self, rows: int, cols: int, spacing: float = 1.0):
        """
        Helper to create a basic 2D lattice with nearest-neighbor connectivity.

        Args:
            rows: Number of rows in the lattice.
            cols: Number of columns in the lattice.
            spacing: Distance between adjacent qubits (uniform).
        """
        try:
            # Assign qubit IDs row-major
            for r in range(rows):
                for c in range(cols):
                    qubit_id = r * cols + c
                    self.add_qubit_position(qubit_id, x=c * spacing, y=r * spacing)

            # Connect nearest neighbors
            for r in range(rows):
                for c in range(cols):
                    qubit_id = r * cols + c
                    # Right neighbor
                    if c + 1 < cols:
                        right_id = r * cols + (c + 1)
                        self.add_connectivity(qubit_id, right_id, distance=spacing)
                    # Down neighbor
                    if r + 1 < rows:
                        down_id = (r + 1) * cols + c
                        self.add_connectivity(qubit_id, down_id, distance=spacing)

            logger.info(f"Built {rows}x{cols} 2D lattice with spacing={spacing}")
        except Exception as e:
            logger.error(f"Error building 2D lattice: {e}")

    def dump_topology(self) -> Dict[str, Any]:
        """
        Return a structured dictionary describing all qubits and links.
        Useful for exporting or debugging.

        Returns:
            Dictionary with keys 'qubits' and 'links'.
        """
        qubits_info = {}
        for qid, pos in self.qubit_positions.items():
            qubits_info[qid] = {
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
            }
        links_info = []
        for link in self.links:
            links_info.append({
                "qubit_a": link.qubit_a,
                "qubit_b": link.qubit_b,
                "distance": link.distance,
                "coupling_strength": link.coupling_strength,
                "error_rate": link.error_rate,
                "metadata": link.metadata,
            })
        return {"qubits": qubits_info, "links": links_info}
