# src/quantum_system/network.py

import logging
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

class NetworkType(Enum):
    """
    Enumerates different types of networks:
      - CLASSICAL: Regular classical communication.
      - QUANTUM: Purely quantum communication channels.
      - HYBRID: Combination of quantum and classical links.
    """
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"

class TransmissionMode(Enum):
    """
    Defines how quantum states or entanglement are transmitted between nodes:
      - DIRECT: Direct physical transmission of qubits or photons.
      - TELEPORTATION: Use of entanglement and classical communication
                      to 'teleport' a quantum state.
    """
    DIRECT = "direct"
    TELEPORTATION = "teleportation"

@dataclass
class NetworkMetrics:
    """
    Tracks performance or status metrics for the network:
      - latency:   Average communication latency (µs).
      - bandwidth: Effective bandwidth (qubits per second or entanglement pairs/sec).
      - fidelity:  Average fidelity of transmitted/entangled states.
      - entanglement_rate: Rate at which entangled pairs are generated (pairs/s).
    """
    latency: float
    bandwidth: float
    fidelity: float
    entanglement_rate: float

class QuantumSwitch:
    """
    Represents a software-defined quantum switch that can create links
    between ports, enabling entanglement establishment. This is a simplified
    model focusing on connectivity and entanglement tracking.
    """

    def __init__(self, num_ports: int):
        """
        Args:
            num_ports: Number of ports on the switch (each port can connect to one node or another switch).
        """
        self.num_ports = num_ports
        # Adjacency matrix for port connectivity
        self.connection_matrix = np.zeros((num_ports, num_ports), dtype=bool)
        # Track whether a port is currently in use
        self.port_states = [False] * num_ports
        # Store pairs of ports that have established entanglement
        self.entangled_pairs: Set[Tuple[int, int]] = set()

        logger.debug(f"QuantumSwitch created with {num_ports} ports.")

    def connect_ports(self, port1: int, port2: int) -> bool:
        """
        Create a direct link between two ports, if both are free.

        Args:
            port1: Index of the first port.
            port2: Index of the second port.

        Returns:
            True if the connection is established, False otherwise.
        """
        if self._validate_ports(port1, port2):
            if not self.port_states[port1] and not self.port_states[port2]:
                self.connection_matrix[port1, port2] = True
                self.connection_matrix[port2, port1] = True
                self.port_states[port1] = True
                self.port_states[port2] = True
                logger.info(f"Ports {port1} and {port2} connected on switch.")
                return True
            else:
                logger.warning(f"Ports {port1} or {port2} already in use.")
        else:
            logger.warning(f"Invalid port indices: {port1}, {port2}.")
        return False

    def establish_entanglement(self, port1: int, port2: int) -> bool:
        """
        Attempt to create an entangled pair between two ports.

        Args:
            port1: Index of the first port.
            port2: Index of the second port.

        Returns:
            True if entanglement is established, False otherwise.
        """
        logger.debug(f"Attempting entanglement between ports {port1} and {port2}.")
        if self.connect_ports(port1, port2):
            pair = tuple(sorted([port1, port2]))
            self.entangled_pairs.add(pair)
            logger.info(f"Entanglement established between ports {pair}.")
            return True
        return False

    def release_connection(self, port1: int, port2: int) -> None:
        """
        Release or break the connection between two ports, freeing them for other uses.

        Args:
            port1: Index of the first port.
            port2: Index of the second port.
        """
        if not self._validate_ports(port1, port2):
            logger.warning(f"Invalid ports {port1}, {port2} for release.")
            return

        if self.connection_matrix[port1, port2]:
            self.connection_matrix[port1, port2] = False
            self.connection_matrix[port2, port1] = False
            self.port_states[port1] = False
            self.port_states[port2] = False

            pair = tuple(sorted([port1, port2]))
            if pair in self.entangled_pairs:
                self.entangled_pairs.remove(pair)
            logger.info(f"Connection released between ports {port1} and {port2}.")

    def _validate_ports(self, *ports: int) -> bool:
        """
        Check that all port indices are within valid range.

        Returns:
            True if all ports are valid, False otherwise.
        """
        return all(0 <= p < self.num_ports for p in ports)

class QuantumNetwork:
    """
    Models a quantum network with multiple nodes and optional quantum switches.
    Tracks basic connectivity, including direct node-to-node links or switch-based
    entanglement distribution.
    """

    def __init__(self, num_nodes: int, network_type: NetworkType = NetworkType.QUANTUM):
        """
        Args:
            num_nodes: Number of quantum nodes in the network.
            network_type: Whether it's a classical, quantum, or hybrid network.
        """
        self.num_nodes = num_nodes
        self.network_type = network_type
        # A list of QuantumSwitches for more complex topologies
        self.switches: List[QuantumSwitch] = []
        # Node-to-node connectivity matrix
        self.node_connections = np.zeros((num_nodes, num_nodes), dtype=bool)
        # Basic network metrics
        self.metrics = NetworkMetrics(
            latency=0.0,
            bandwidth=0.0,
            fidelity=1.0,
            entanglement_rate=0.0
        )

        logger.debug(f"QuantumNetwork created with {num_nodes} nodes, type={network_type.value}.")

    def add_switch(self, num_ports: int) -> int:
        """
        Add a quantum switch with a specified number of ports, returning its index.

        Args:
            num_ports: Number of ports on the newly created quantum switch.

        Returns:
            The index (in self.switches) of the newly added switch.
        """
        switch = QuantumSwitch(num_ports)
        self.switches.append(switch)
        switch_id = len(self.switches) - 1
        logger.info(f"Added QuantumSwitch with ID={switch_id}, ports={num_ports}.")
        return switch_id

    def connect_nodes(self, 
                      node1: int, 
                      node2: int, 
                      mode: TransmissionMode = TransmissionMode.DIRECT) -> bool:
        """
        Connect two quantum nodes, possibly with a specified transmission mode
        (direct fiber or teleportation-based).
        
        Args:
            node1: Index of the first node.
            node2: Index of the second node.
            mode: TransmissionMode (DIRECT or TELEPORTATION).

        Returns:
            True if connection is established, False otherwise.
        """
        if not (0 <= node1 < self.num_nodes and 0 <= node2 < self.num_nodes):
            logger.warning(f"Invalid node indices: {node1}, {node2}.")
            return False

        # For demonstration, we only toggle a direct adjacency matrix.
        # Teleportation might require entanglement distribution via a switch, etc.
        self.node_connections[node1, node2] = True
        self.node_connections[node2, node1] = True
        logger.info(f"Nodes {node1} and {node2} connected via {mode.value} transmission.")
        return True

    def get_network_metrics(self) -> Dict[str, float]:
        """
        Retrieve current network performance metrics.

        Returns:
            A dictionary summarizing latency, bandwidth, fidelity, and entanglement rate.
        """
        # In a real system, you'd measure or calculate these dynamically
        return {
            "latency_us": self.metrics.latency,
            "bandwidth_qps": self.metrics.bandwidth,  # qubits per second
            "fidelity": self.metrics.fidelity,
            "entanglement_rate_hz": self.metrics.entanglement_rate
        }

    def update_metrics(self, 
                       latency: float = None,
                       bandwidth: float = None,
                       fidelity: float = None,
                       entanglement_rate: float = None) -> None:
        """
        Update the network metrics. Typically called as conditions change 
        (e.g., load increases, better entanglement operations, etc.).

        Args:
            latency: New average latency (µs).
            bandwidth: New bandwidth (qubits/sec).
            fidelity: Average fidelity of transmissions/entanglement.
            entanglement_rate: Updated entanglement generation rate (pairs/sec).
        """
        if latency is not None:
            self.metrics.latency = latency
        if bandwidth is not None:
            self.metrics.bandwidth = bandwidth
        if fidelity is not None:
            self.metrics.fidelity = fidelity
        if entanglement_rate is not None:
            self.metrics.entanglement_rate = entanglement_rate
        
        logger.debug(
            "Network metrics updated: "
            f"latency={self.metrics.latency}, "
            f"bandwidth={self.metrics.bandwidth}, "
            f"fidelity={self.metrics.fidelity}, "
            f"entanglement_rate={self.metrics.entanglement_rate}."
        )

    def route_via_switch(self, 
                         switch_id: int, 
                         port_in: int,
                         port_out: int,
                         node_in: int,
                         node_out: int) -> bool:
        """
        Demonstration: show how you might route data/entanglement through a specific switch,
        connecting node_in to node_out via switch ports.

        Args:
            switch_id: The index of the target QuantumSwitch in self.switches.
            port_in: Switch port connected to node_in.
            port_out: Switch port connected to node_out.
            node_in: Index of input node.
            node_out: Index of output node.

        Returns:
            True if routing was successful, False otherwise.
        """
        if switch_id < 0 or switch_id >= len(self.switches):
            logger.warning(f"Invalid switch ID {switch_id}.")
            return False

        if not (0 <= node_in < self.num_nodes and 0 <= node_out < self.num_nodes):
            logger.warning(f"Invalid node indices for routing: {node_in}, {node_out}.")
            return False

        qs = self.switches[switch_id]
        # Attempt to connect the ports if free
        success = qs.connect_ports(port_in, port_out)
        if success:
            logger.info(f"Node {node_in} is now routed to Node {node_out} via Switch {switch_id}, ports {port_in}-{port_out}.")
        else:
            logger.warning(f"Failed to route via Switch {switch_id}, ports {port_in}-{port_out}.")

        return success
