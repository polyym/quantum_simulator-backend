# src/quantum_system/network.py

from enum import Enum
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import numpy as np

class NetworkType(Enum):
    """Network types from paper"""
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"

class TransmissionMode(Enum):
    """Quantum state transmission modes from paper"""
    DIRECT = "direct"
    TELEPORTATION = "teleportation"

@dataclass
class NetworkMetrics:
    """Performance metrics for quantum network"""
    latency: float             # Communication latency
    bandwidth: float           # Available bandwidth
    fidelity: float           # Quantum state fidelity
    entanglement_rate: float  # Rate of entanglement generation

class QuantumSwitch:
    """Software-defined quantum network switch from paper"""
    def __init__(self, num_ports: int):
        self.num_ports = num_ports
        self.connection_matrix = np.zeros((num_ports, num_ports), dtype=bool)
        self.port_states = [False] * num_ports  # Track busy ports
        self.entangled_pairs: Set[tuple] = set()
        
    def connect_ports(self, port1: int, port2: int) -> bool:
        """Establish quantum connection between ports"""
        if self._validate_ports(port1, port2):
            if not self.port_states[port1] and not self.port_states[port2]:
                self.connection_matrix[port1, port2] = True
                self.connection_matrix[port2, port1] = True
                self.port_states[port1] = True
                self.port_states[port2] = True
                return True
        return False
        
    def establish_entanglement(self, port1: int, port2: int) -> bool:
        """Create entanglement between ports"""
        if self.connect_ports(port1, port2):
            self.entangled_pairs.add(tuple(sorted([port1, port2])))
            return True
        return False
        
    def _validate_ports(self, *ports) -> bool:
        """Validate port numbers"""
        return all(0 <= p < self.num_ports for p in ports)

class QuantumNetwork:
    """Quantum network implementation from paper"""
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.switches = []
        self.node_connections = np.zeros((num_nodes, num_nodes), dtype=bool)
        self.metrics = NetworkMetrics(0.0, 0.0, 1.0, 0.0)
        
    def add_switch(self, num_ports: int) -> int:
        """Add quantum switch to network"""
        switch = QuantumSwitch(num_ports)
        self.switches.append(switch)
        return len(self.switches) - 1
        
    def connect_nodes(self, node1: int, node2: int, 
                     mode: TransmissionMode = TransmissionMode.DIRECT) -> bool:
        """Connect quantum nodes with specified transmission mode"""
        if 0 <= node1 < self.num_nodes and 0 <= node2 < self.num_nodes:
            self.node_connections[node1, node2] = True
            self.node_connections[node2, node1] = True
            return True
        return False
        
    def get_network_metrics(self) -> Dict[str, float]:
        """Get current network performance metrics"""
        return {
            'latency_us': self.metrics.latency,
            'bandwidth_qbits': self.metrics.bandwidth,
            'fidelity': self.metrics.fidelity,
            'entanglement_rate_hz': self.metrics.entanglement_rate
        }