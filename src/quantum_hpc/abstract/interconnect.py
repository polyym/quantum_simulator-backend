# src/quantum_hpc/abstract/interconnect.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

@dataclass
class LinkConfig:
    """
    Configuration for a single interconnect link between two quantum processors or nodes.
    
    Attributes:
        node_a: Identifier of the first node (string, int, or other).
        node_b: Identifier of the second node.
        distance: Physical or logical distance between the nodes (for modeling latency or attenuation).
        bandwidth: Bandwidth or maximum entanglement rate, in pairs per second or similar metric.
        loss_rate: Probability of link loss or attenuation factor for photonic/optical setups.
        metadata: Arbitrary additional link data (e.g., device type, fiber length).
    """
    node_a: Any
    node_b: Any
    distance: float = 1.0
    bandwidth: float = 1.0
    loss_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class Interconnect(ABC):
    """
    Abstract base class for quantum interconnects between multiple processors or nodes.

    Derived classes should implement methods for:
      1. Initializing or configuring links between nodes.
      2. Generating or distributing entanglement across links.
      3. Handling time delays, bandwidth limits, or error/loss models.
      4. Providing link status or metrics (entanglement success rate, usage stats).
    """

    @abstractmethod
    def add_link(self, link_config: LinkConfig) -> None:
        """
        Add or configure a link between two nodes.

        Args:
            link_config: Configuration describing the link’s properties (distance, bandwidth, etc.)
        """
        pass

    @abstractmethod
    def remove_link(self, node_a: Any, node_b: Any) -> None:
        """
        Remove or disable a link between two nodes.
        
        Args:
            node_a: Identifier for the first node.
            node_b: Identifier for the second node.
        """
        pass

    @abstractmethod
    def generate_entanglement(self, node_a: Any, node_b: Any, *args, **kwargs) -> bool:
        """
        Attempt to generate or distribute entanglement between two nodes.

        Args:
            node_a: Identifier for the first node.
            node_b: Identifier for the second node.
            *args, **kwargs: Additional parameters or protocols (e.g., EPR pair generation).

        Returns:
            True if entanglement succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def get_link_status(self, node_a: Any, node_b: Any) -> Dict[str, Any]:
        """
        Retrieve the current status or metrics of the link (bandwidth usage, success rate, etc.)

        Args:
            node_a: Identifier for the first node.
            node_b: Identifier for the second node.

        Returns:
            Dictionary containing relevant status fields (e.g., 'bandwidth_used', 'loss_rate', 'uptime').
        """
        pass

#
# Example Concrete Class
#

class BasicQuantumInterconnect(Interconnect):
    """
    A simple, production-ready interconnect that models basic entanglement generation
    with a fixed success probability derived from link distance and loss_rate.
    """

    def __init__(self):
        self.links: Dict[Tuple[Any, Any], LinkConfig] = {}
        logger.debug("BasicQuantumInterconnect created.")

    def add_link(self, link_config: LinkConfig) -> None:
        # Ensure consistent ordering of node identifiers (a < b) to store in dict
        key = self._make_key(link_config.node_a, link_config.node_b)
        self.links[key] = link_config
        logger.info(f"Link added between {link_config.node_a} and {link_config.node_b} with config: {link_config}.")

    def remove_link(self, node_a: Any, node_b: Any) -> None:
        key = self._make_key(node_a, node_b)
        if key in self.links:
            del self.links[key]
            logger.info(f"Link removed between {node_a} and {node_b}.")
        else:
            logger.warning(f"Cannot remove link between {node_a} and {node_b}; link not found.")

    def generate_entanglement(self, node_a: Any, node_b: Any, *args, **kwargs) -> bool:
        key = self._make_key(node_a, node_b)
        if key not in self.links:
            logger.error(f"No link found between {node_a} and {node_b} for entanglement.")
            return False
        
        link_cfg = self.links[key]
        # Example success probability: a function of distance, bandwidth, and loss_rate
        base_success = 1.0 - link_cfg.loss_rate
        # Possibly scale success by distance or incorporate bandwidth usage
        distance_factor = max(0.0, 1.0 - 0.01 * link_cfg.distance)
        success_prob = base_success * distance_factor

        outcome = (kwargs.get('force_success', False) or
                   (success_prob > 0.0 and (success_prob > 1.0 or success_prob > kwargs.get('rand_val', 0.0))))

        logger.debug(f"Entanglement attempt between {node_a} and {node_b} => p={success_prob:.3f}, outcome={outcome}")
        return outcome

    def get_link_status(self, node_a: Any, node_b: Any) -> Dict[str, Any]:
        key = self._make_key(node_a, node_b)
        if key not in self.links:
            return {"error": "link_not_found"}
        
        link_cfg = self.links[key]
        return {
            "distance": link_cfg.distance,
            "bandwidth": link_cfg.bandwidth,
            "loss_rate": link_cfg.loss_rate,
            "metadata": link_cfg.metadata
        }

    def _make_key(self, a: Any, b: Any) -> Tuple[Any, Any]:
        """
        Helper to generate a sorted tuple for dict storage.
        """
        return (a, b) if str(a) < str(b) else (b, a)
