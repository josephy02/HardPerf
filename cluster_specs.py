import logging
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

from gpu_specs import get_gpu_spec, GPUSpec

logger = logging.getLogger(__name__)

class InterconnectType(Enum):
    """Types of GPU interconnects."""
    NVLINK = "nvlink"
    NVSWITCH = "nvswitch"
    INFINIBAND = "infiniband"
    ETHERNET = "ethernet"
    PCIE = "pcie"

@dataclass
class InterconnectSpec:
    """Interconnect specification with bandwidth characteristics."""
    type: InterconnectType
    bandwidth_gb_s: float
    latency_us: float
    topology: str  # "mesh", "tree", "torus", "fat_tree"

    def get_effective_bandwidth(self, num_gpus: int) -> float:
        """Calculate effective bandwidth accounting for contention."""
        if self.type == InterconnectType.NVLINK:
            # NVLink up to 8 GPUs
            if num_gpus <= 8:
                return self.bandwidth_gb_s * 0.85
            else:
                return self.bandwidth_gb_s * 0.70

        elif self.type == InterconnectType.NVSWITCH:
            # NVSwitch maintains high bandwidth across larger clusters
            if num_gpus <= 16:
                return self.bandwidth_gb_s * 0.90
            else:
                return self.bandwidth_gb_s * 0.80

        elif self.type == InterconnectType.INFINIBAND:
            # InfiniBand scaling depends on switch architecture
            if num_gpus <= 32:
                return self.bandwidth_gb_s * 0.75
            else:
                return self.bandwidth_gb_s * 0.60

        else:  # Ethernet, PCIe
            # Lower performance interconnects
            return self.bandwidth_gb_s * 0.50

# Predefined interconnect specifications
INTERCONNECT_SPECS = {
    "nvlink_4": InterconnectSpec(
        type=InterconnectType.NVLINK,
        bandwidth_gb_s=600.0,  # NVLink 4.0
        latency_us=1.0,
        topology="mesh"
    ),

    "nvlink_5": InterconnectSpec(
        type=InterconnectType.NVLINK,
        bandwidth_gb_s=900.0,  # NVLink 5.0 (H100)
        latency_us=0.8,
        topology="mesh"
    ),

    "nvswitch_3": InterconnectSpec(
        type=InterconnectType.NVSWITCH,
        bandwidth_gb_s=900.0,  # NVSwitch 3.0
        latency_us=1.2,
        topology="fat_tree"
    ),

    "infiniband_hdr": InterconnectSpec(
        type=InterconnectType.INFINIBAND,
        bandwidth_gb_s=25.0,   # 200 Gbps HDR
        latency_us=2.0,
        topology="fat_tree"
    ),

    "infiniband_ndr": InterconnectSpec(
        type=InterconnectType.INFINIBAND,
        bandwidth_gb_s=50.0,   # 400 Gbps NDR
        latency_us=1.8,
        topology="fat_tree"
    ),

    "ethernet_100g": InterconnectSpec(
        type=InterconnectType.ETHERNET,
        bandwidth_gb_s=12.5,   # 100 Gbps Ethernet
        latency_us=10.0,
        topology="tree"
    )
}

@dataclass
class ClusterSpec:
    """Multi-GPU cluster specification."""
    num_gpus: int
    gpu_type: str
    interconnect: str = "nvlink_5"
    nodes: int = 1
    gpus_per_node: int = None

    def __post_init__(self):
        """Initialize derived properties."""
        self.gpu_spec = get_gpu_spec(self.gpu_type)
        self.interconnect_spec = INTERCONNECT_SPECS.get(
            self.interconnect, INTERCONNECT_SPECS["nvlink_5"]
        )

        # Calculate nodes and GPUs per node
        if self.gpus_per_node is None:
            self.gpus_per_node = min(8, self.num_gpus)  # Typical node limit

        if self.nodes == 1:
            self.nodes = max(1, (self.num_gpus + self.gpus_per_node - 1) // self.gpus_per_node)

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate cluster configuration."""
        if self.num_gpus <= 0:
            raise ValueError(f"Invalid number of GPUs: {self.num_gpus}")

        if self.nodes * self.gpus_per_node < self.num_gpus:
            logger.warning(f"Cluster configuration mismatch: {self.nodes} nodes × "
                         f"{self.gpus_per_node} GPUs/node < {self.num_gpus} total GPUs")

        # Check for optimal configurations
        if self.num_gpus > 8 and self.interconnect.startswith("nvlink"):
            logger.warning("NVLink typically optimal for ≤8 GPUs. Consider NVSwitch for larger clusters.")

    def get_total_memory_capacity(self) -> float:
        """Get total cluster memory capacity in GB."""
        return self.num_gpus * self.gpu_spec.memory_capacity_gb

    def get_effective_bandwidth(self) -> float:
        """Get effective interconnect bandwidth for this cluster size."""
        return self.interconnect_spec.get_effective_bandwidth(self.num_gpus)

    def estimate_communication_latency(self) -> float:
        """Estimate average communication latency in microseconds."""
        base_latency = self.interconnect_spec.latency_us

        # Add topology penalty for multi-node
        if self.nodes > 1:
            base_latency *= 1.5  # Inter-node penalty

        # Add contention penalty for large clusters
        if self.num_gpus > 16:
            base_latency *= 1.2

        return base_latency

    def get_topology_info(self) -> Dict:
        """Get cluster topology information."""
        return {
            "total_gpus": self.num_gpus,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "gpu_type": self.gpu_type,
            "interconnect": self.interconnect,
            "topology": self.interconnect_spec.topology,
            "total_memory_gb": self.get_total_memory_capacity(),
            "effective_bandwidth_gb_s": self.get_effective_bandwidth(),
            "estimated_latency_us": self.estimate_communication_latency()
        }

    def recommend_optimal_interconnect(self) -> str:
        """Recommend optimal interconnect for this cluster size."""
        if self.num_gpus <= 8:
            return "nvlink_5"
        elif self.num_gpus <= 32:
            return "nvswitch_3"
        elif self.num_gpus <= 128:
            return "infiniband_ndr"
        else:
            return "infiniband_ndr"

    def calculate_power_consumption(self) -> Dict:
        """Estimate cluster power consumption."""
        # Approximate GPU power consumption (watts)
        gpu_power_map = {
            "NVIDIA H100": 700,
            "NVIDIA H200": 700,
            "NVIDIA B200": 1000,  # Estimated
            "NVIDIA B300": 1200,  # Estimated
            "NVIDIA A100": 400,
            "AMD MI300X": 750,
            "AMD MI300A": 750,
            "AMD MI250X": 500,
            "AMD MI250": 500
        }

        gpu_power = gpu_power_map.get(self.gpu_type, 400)  # Default 400W

        # Add system overhead (CPU, memory, cooling, etc.)
        system_overhead_per_gpu = 100  # watts

        total_power_w = self.num_gpus * (gpu_power + system_overhead_per_gpu)

        return {
            "gpu_power_per_device_w": gpu_power,
            "system_overhead_per_gpu_w": system_overhead_per_gpu,
            "total_cluster_power_w": total_power_w,
            "total_cluster_power_kw": total_power_w / 1000,
            "estimated_monthly_cost_usd": total_power_w * 24 * 30 * 0.10 / 1000  # $0.10/kWh
        }

def create_cluster_spec(num_gpus: int, gpu_type: str = "NVIDIA H100",
                       interconnect: str = None, nodes: int = None) -> ClusterSpec:
    """Factory function to create optimal cluster specification.

    Args:
        num_gpus: Number of GPUs in cluster
        gpu_type: GPU type name
        interconnect: Interconnect type (auto-selected if None)
        nodes: Number of nodes (auto-calculated if None)

    Returns:
        ClusterSpec: Optimized cluster specification
    """
    # Auto-select interconnect if not specified
    if interconnect is None:
        if num_gpus <= 8:
            interconnect = "nvlink_5"
        elif num_gpus <= 32:
            interconnect = "nvswitch_3"
        else:
            interconnect = "infiniband_ndr"

    # Auto-calculate nodes if not specified
    if nodes is None:
        if num_gpus <= 8:
            nodes = 1
        else:
            nodes = max(1, (num_gpus + 7) // 8)  # 8 GPUs per node typical

    return ClusterSpec(
        num_gpus=num_gpus,
        gpu_type=gpu_type,
        interconnect=interconnect,
        nodes=nodes
    )

def compare_cluster_configurations(configs: List[ClusterSpec]) -> Dict:
    """Compare multiple cluster configurations.

    Args:
        configs: List of cluster specifications to compare

    Returns:
        Dict: Comparison results
    """
    comparison = {}

    for i, config in enumerate(configs):
        name = f"config_{i+1}"
        topo_info = config.get_topology_info()
        power_info = config.calculate_power_consumption()

        comparison[name] = {
            **topo_info,
            "power_consumption_kw": power_info["total_cluster_power_kw"],
            "estimated_monthly_cost_usd": power_info["estimated_monthly_cost_usd"],
            "memory_per_gpu_gb": config.gpu_spec.memory_capacity_gb,
            "bandwidth_per_gpu_gb_s": config.gpu_spec.memory_bandwidth_gb_s
        }

    return comparison

def get_recommended_cluster_size(model_size_billions: float,
                                target_batch_size: int = 32,
                                strategy: str = "tensor_parallel") -> Dict:
    """Recommend cluster size for given model requirements.

    Args:
        model_size_billions: Model size in billions of parameters
        target_batch_size: Desired batch size
        strategy: Parallelization strategy

    Returns:
        Dict: Cluster size recommendations
    """
    # Rough memory estimates (will be refined by actual analysis)
    estimated_memory_per_gpu = model_size_billions * 4  # Conservative estimate

    recommendations = {}

    # Try different GPU types
    gpu_types = ["NVIDIA H100", "NVIDIA H200", "NVIDIA B200", "AMD MI300X"]

    for gpu_type in gpu_types:
        try:
            gpu_spec = get_gpu_spec(gpu_type)
            gpu_memory = gpu_spec.memory_capacity_gb

            if strategy == "data_parallel":
                # Each GPU needs full model
                if estimated_memory_per_gpu <= gpu_memory * 0.9:
                    min_gpus = max(1, (target_batch_size + 7) // 8)  # 8 batch per GPU typical
                    recommendations[gpu_type] = {
                        "min_gpus": min_gpus,
                        "strategy": "data_parallel",
                        "memory_per_gpu_gb": estimated_memory_per_gpu,
                        "feasible": True
                    }
                else:
                    recommendations[gpu_type] = {"feasible": False, "reason": "Model too large"}

            elif strategy == "tensor_parallel":
                # Model memory scales down with GPU count
                min_gpus = max(1, int(estimated_memory_per_gpu / (gpu_memory * 0.8)))
                recommendations[gpu_type] = {
                    "min_gpus": min_gpus,
                    "strategy": "tensor_parallel",
                    "memory_per_gpu_gb": estimated_memory_per_gpu / min_gpus,
                    "feasible": True
                }

        except Exception as e:
            logger.warning(f"Could not analyze {gpu_type}: {e}")

    return recommendations