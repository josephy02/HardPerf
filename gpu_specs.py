"""
Author              :
Copyright           :
File Name           : gpu_specs.py of memprof-training, Memory Profiler under ML Training
Description         : GPU Hardware Specifications
                      Centralized hardware data for memory performance analysis
                      Enhanced with multi-GPU and interconnect support


Revision History    :
Date                  Author               Comments
--------------------------------------------------------------------------------------------------

"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class GPUSpec:
    """GPU hardware specification."""
    name: str
    memory_capacity_gb: float
    memory_bandwidth_gb_s: float
    memory_type: str
    compute_capability: str
    tensor_cores: bool
    nvlink_bandwidth_gb_s: Optional[float] = None
    memory_bus_width: int = 0
    memory_clock_mhz: float = 0
    max_gpus_per_node: int = 8  # Typical maximum GPUs per node
    supports_nvswitch: bool = False  # Whether GPU supports NVSwitch

    def __post_init__(self):
        """Calculate derived properties."""
        # Estimate effective bandwidth (typically 85-90% of theoretical)
        if self.memory_type.startswith("HBM"):
            self.effective_bandwidth_gb_s = self.memory_bandwidth_gb_s * 0.90
        elif self.memory_type.startswith("GDDR"):
            self.effective_bandwidth_gb_s = self.memory_bandwidth_gb_s * 0.85
        else:
            self.effective_bandwidth_gb_s = self.memory_bandwidth_gb_s * 0.80

# GPU Hardware Database
GPU_SPECS: Dict[str, GPUSpec] = {
    # NVIDIA B-Series (Blackwell - Next Generation)
    "NVIDIA B200": GPUSpec(
        name="NVIDIA B200",
        memory_capacity_gb=192.0,
        memory_bandwidth_gb_s=8000.0,
        memory_type="HBM3e",
        compute_capability="10.0",
        tensor_cores=True,
        nvlink_bandwidth_gb_s=1800.0,
        memory_bus_width=6144,
        memory_clock_mhz=5200,
        max_gpus_per_node=8,
        supports_nvswitch=True
    ),

    # Note: B300 specs are projected/estimated based on expected progression
    "NVIDIA B300": GPUSpec(
        name="NVIDIA B300",
        memory_capacity_gb=256.0,
        memory_bandwidth_gb_s=10000.0,
        memory_type="HBM4",
        compute_capability="10.0",
        tensor_cores=True,
        nvlink_bandwidth_gb_s=2400.0,
        memory_bus_width=8192,
        memory_clock_mhz=6000,
        max_gpus_per_node=8,
        supports_nvswitch=True
    ),

    # NVIDIA H-Series (Hopper)
    "NVIDIA H100": GPUSpec(
        name="NVIDIA H100",
        memory_capacity_gb=80.0,
        memory_bandwidth_gb_s=3352.0,
        memory_type="HBM3",
        compute_capability="9.0",
        tensor_cores=True,
        nvlink_bandwidth_gb_s=900.0,
        memory_bus_width=5120,
        memory_clock_mhz=2619,
        max_gpus_per_node=8,
        supports_nvswitch=True
    ),

    "NVIDIA H200": GPUSpec(
        name="NVIDIA H200",
        memory_capacity_gb=141.0,
        memory_bandwidth_gb_s=4800.0,
        memory_type="HBM3e",
        compute_capability="9.0",
        tensor_cores=True,
        nvlink_bandwidth_gb_s=900.0,
        memory_bus_width=5120,
        memory_clock_mhz=4800,
        max_gpus_per_node=8,
        supports_nvswitch=True
    ),

    # NVIDIA A-Series (Ampere)
    "NVIDIA A100": GPUSpec(
        name="NVIDIA A100",
        memory_capacity_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        memory_type="HBM2e",
        compute_capability="8.0",
        tensor_cores=True,
        nvlink_bandwidth_gb_s=600.0,
        memory_bus_width=5120,
        memory_clock_mhz=1593,
        max_gpus_per_node=8,
        supports_nvswitch=True
    ),

    "NVIDIA A100 40GB": GPUSpec(
        name="NVIDIA A100 40GB",
        memory_capacity_gb=40.0,
        memory_bandwidth_gb_s=1555.0,
        memory_type="HBM2",
        compute_capability="8.0",
        tensor_cores=True,
        nvlink_bandwidth_gb_s=600.0,
        memory_bus_width=5120,
        memory_clock_mhz=1215,
        max_gpus_per_node=8,
        supports_nvswitch=True
    ),

    # NVIDIA L-Series (Data Center)
    "NVIDIA L40S": GPUSpec(
        name="NVIDIA L40S",
        memory_capacity_gb=48.0,
        memory_bandwidth_gb_s=864.0,
        memory_type="GDDR6",
        compute_capability="8.9",
        tensor_cores=True,
        memory_bus_width=384,
        memory_clock_mhz=18000,
        max_gpus_per_node=4,  # Typically fewer L40S per node
        supports_nvswitch=False
    ),

    "NVIDIA L40": GPUSpec(
        name="NVIDIA L40",
        memory_capacity_gb=48.0,
        memory_bandwidth_gb_s=696.0,
        memory_type="GDDR6",
        compute_capability="8.9",
        tensor_cores=True,
        memory_bus_width=384,
        memory_clock_mhz=14500,
        max_gpus_per_node=4,
        supports_nvswitch=False
    ),

    # AMD MI Series
    "AMD MI300X": GPUSpec(
        name="AMD MI300X",
        memory_capacity_gb=192.0,
        memory_bandwidth_gb_s=5200.0,
        memory_type="HBM3",
        compute_capability="gfx940",
        tensor_cores=True,
        memory_bus_width=8192,
        memory_clock_mhz=1300,
        max_gpus_per_node=8,
        supports_nvswitch=False  # Uses AMD Infinity Fabric instead
    ),

    "AMD MI300A": GPUSpec(
        name="AMD MI300A",
        memory_capacity_gb=128.0,
        memory_bandwidth_gb_s=5200.0,
        memory_type="HBM3",
        compute_capability="gfx940",
        tensor_cores=True,
        memory_bus_width=8192,
        memory_clock_mhz=1300,
        max_gpus_per_node=8,
        supports_nvswitch=False
    ),

    "AMD MI250X": GPUSpec(
        name="AMD MI250X",
        memory_capacity_gb=128.0,
        memory_bandwidth_gb_s=3276.0,
        memory_type="HBM2e",
        compute_capability="gfx90a",
        tensor_cores=True,
        memory_bus_width=8192,
        memory_clock_mhz=1600,
        max_gpus_per_node=8,
        supports_nvswitch=False
    ),

    "AMD MI250": GPUSpec(
        name="AMD MI250",
        memory_capacity_gb=128.0,
        memory_bandwidth_gb_s=3276.0,
        memory_type="HBM2e",
        compute_capability="gfx90a",
        tensor_cores=True,
        memory_bus_width=8192,
        memory_clock_mhz=1600,
        max_gpus_per_node=8,
        supports_nvswitch=False
    ),
}

# Legacy aliases for backward compatibility
GPU_ALIASES = {
    # NVIDIA aliases
    "B200": "NVIDIA B200",
    "B300": "NVIDIA B300",
    "H100": "NVIDIA H100",
    "H200": "NVIDIA H200",
    "A100": "NVIDIA A100",
    "L40S": "NVIDIA L40S",
    "L40": "NVIDIA L40",

    # AMD aliases
    "MI300X": "AMD MI300X",
    "MI300A": "AMD MI300A",
    "MI250X": "AMD MI250X",
    "MI250": "AMD MI250",
}

def get_gpu_spec(gpu_type: str) -> GPUSpec:
    """Get GPU specification by name.

    Args:
        gpu_type: GPU type name

    Returns:
        GPUSpec: GPU specification

    Raises:
        ValueError: If GPU type not found
    """
    # Check direct match first
    if gpu_type in GPU_SPECS:
        return GPU_SPECS[gpu_type]

    # Check aliases
    if gpu_type in GPU_ALIASES:
        return GPU_SPECS[GPU_ALIASES[gpu_type]]

    # Partial matching for flexibility
    gpu_lower = gpu_type.lower()
    for spec_name, spec in GPU_SPECS.items():
        if gpu_lower in spec_name.lower():
            logger.info(f"Matched '{gpu_type}' to '{spec_name}'")
            return spec

    # Default fallback
    logger.warning(f"GPU type '{gpu_type}' not found, using NVIDIA H100 as default")
    return GPU_SPECS["NVIDIA H100"]

def get_gpu_memory_capacity(gpu_type: str) -> float:
    """Get GPU memory capacity in GB."""
    return get_gpu_spec(gpu_type).memory_capacity_gb

def get_gpu_bandwidth(gpu_type: str) -> float:
    """Get GPU memory bandwidth in GB/s."""
    return get_gpu_spec(gpu_type).memory_bandwidth_gb_s

def get_gpu_effective_bandwidth(gpu_type: str) -> float:
    """Get GPU effective memory bandwidth in GB/s."""
    return get_gpu_spec(gpu_type).effective_bandwidth_gb_s

def get_gpu_nvlink_bandwidth(gpu_type: str) -> Optional[float]:
    """Get GPU NVLink bandwidth in GB/s."""
    return get_gpu_spec(gpu_type).nvlink_bandwidth_gb_s

def get_max_gpus_per_node(gpu_type: str) -> int:
    """Get maximum GPUs per node for given GPU type."""
    return get_gpu_spec(gpu_type).max_gpus_per_node

def supports_nvswitch(gpu_type: str) -> bool:
    """Check if GPU supports NVSwitch interconnect."""
    return get_gpu_spec(gpu_type).supports_nvswitch

def list_supported_gpus() -> list:
    """List all supported GPU types."""
    return list(GPU_SPECS.keys()) + list(GPU_ALIASES.keys())

def get_gpus_by_vendor() -> Dict[str, list]:
    """Get GPUs organized by vendor."""
    vendors = {"NVIDIA": [], "AMD": []}

    for gpu_name in GPU_SPECS.keys():
        if gpu_name.startswith("NVIDIA"):
            vendors["NVIDIA"].append(gpu_name)
        elif gpu_name.startswith("AMD"):
            vendors["AMD"].append(gpu_name)

    return vendors

def get_latest_gpus() -> Dict[str, str]:
    """Get the latest/most powerful GPU from each vendor."""
    return {
        "NVIDIA": "NVIDIA B300",  # Most advanced
        "AMD": "AMD MI300X"
    }

def get_multi_gpu_capable_gpus() -> List[str]:
    """Get list of GPUs suitable for multi-GPU training."""
    multi_gpu_gpus = []

    for gpu_name, spec in GPU_SPECS.items():
        # Include GPUs with NVLink/high bandwidth interconnect
        if (spec.nvlink_bandwidth_gb_s and spec.nvlink_bandwidth_gb_s > 400) or \
           spec.supports_nvswitch or \
           spec.memory_capacity_gb >= 80:  # High-memory GPUs typically used in clusters
            multi_gpu_gpus.append(gpu_name)

    return multi_gpu_gpus

def estimate_optimal_cluster_size(gpu_type: str, model_size_billions: float) -> Dict:
    """Estimate optimal cluster configuration for given model size.

    Args:
        gpu_type: GPU type name
        model_size_billions: Model size in billions of parameters

    Returns:
        Dict: Cluster size recommendations
    """
    spec = get_gpu_spec(gpu_type)

    # Rough memory estimate: 4GB per billion parameters (conservative)
    estimated_memory_gb = model_size_billions * 4

    # Single GPU capacity check
    if estimated_memory_gb <= spec.memory_capacity_gb * 0.8:
        return {
            "recommended_gpus": 1,
            "strategy": "single_gpu",
            "reason": "Model fits on single GPU",
            "memory_utilization": estimated_memory_gb / spec.memory_capacity_gb
        }

    # Data parallel recommendation
    data_parallel_gpus = max(2, min(8, int(estimated_memory_gb / (spec.memory_capacity_gb * 0.8))))

    # Tensor parallel recommendation (for memory reduction)
    tensor_parallel_gpus = max(2, int(estimated_memory_gb / (spec.memory_capacity_gb * 0.6)))
    tensor_parallel_gpus = min(tensor_parallel_gpus, spec.max_gpus_per_node)

    recommendations = {
        "model_size_billions": model_size_billions,
        "estimated_memory_gb": estimated_memory_gb,
        "gpu_memory_capacity_gb": spec.memory_capacity_gb,
        "strategies": {
            "data_parallel": {
                "gpus": data_parallel_gpus,
                "memory_per_gpu_gb": estimated_memory_gb,
                "feasible": estimated_memory_gb <= spec.memory_capacity_gb * 0.8
            },
            "tensor_parallel": {
                "gpus": tensor_parallel_gpus,
                "memory_per_gpu_gb": estimated_memory_gb / tensor_parallel_gpus,
                "feasible": True
            }
        }
    }

    # Determine best recommendation
    if recommendations["strategies"]["data_parallel"]["feasible"]:
        recommendations["recommended_strategy"] = "data_parallel"
        recommendations["recommended_gpus"] = data_parallel_gpus
    else:
        recommendations["recommended_strategy"] = "tensor_parallel"
        recommendations["recommended_gpus"] = tensor_parallel_gpus

    return recommendations

def compare_gpus(gpu_types: list) -> Dict[str, Dict]:
    """Compare specifications of multiple GPUs.

    Args:
        gpu_types: List of GPU type names

    Returns:
        Dict: Comparison data for each GPU
    """
    comparison = {}

    for gpu_type in gpu_types:
        try:
            spec = get_gpu_spec(gpu_type)
            comparison[gpu_type] = {
                "memory_gb": spec.memory_capacity_gb,
                "bandwidth_gb_s": spec.memory_bandwidth_gb_s,
                "memory_type": spec.memory_type,
                "tensor_cores": spec.tensor_cores,
                "compute_capability": spec.compute_capability,
                "nvlink_bandwidth_gb_s": spec.nvlink_bandwidth_gb_s,
                "max_gpus_per_node": spec.max_gpus_per_node,
                "supports_nvswitch": spec.supports_nvswitch
            }
        except Exception as e:
            logger.warning(f"Could not get specs for {gpu_type}: {e}")
            comparison[gpu_type] = None

    return comparison

def recommend_gpu_for_model(model_size_billions: float, batch_size: int = 32,
                           budget_tier: str = "high", multi_gpu: bool = False) -> Dict[str, str]:
    """Recommend GPUs based on model requirements and budget.

    Args:
        model_size_billions: Model size in billions of parameters
        batch_size: Desired batch size
        budget_tier: Budget tier ("high", "medium", "low")
        multi_gpu: Whether to consider multi-GPU setups

    Returns:
        Dict: Recommended GPUs by category
    """
    # Estimate memory requirements (rough calculation)
    estimated_memory_gb = model_size_billions * 4  # Conservative estimate

    recommendations = {
        "optimal": None,
        "budget": None,
        "enterprise": None
    }

    if budget_tier == "high":
        if estimated_memory_gb > 150 or multi_gpu:
            recommendations["optimal"] = "NVIDIA B300"
            recommendations["enterprise"] = "AMD MI300X"
        elif estimated_memory_gb > 80:
            recommendations["optimal"] = "NVIDIA B200"
            recommendations["enterprise"] = "NVIDIA H200"
        else:
            recommendations["optimal"] = "NVIDIA H100"
            recommendations["enterprise"] = "NVIDIA A100"

    elif budget_tier == "medium":
        if estimated_memory_gb > 100:
            recommendations["optimal"] = "AMD MI300X"
            recommendations["budget"] = "AMD MI250X"
        else:
            recommendations["optimal"] = "NVIDIA A100"
            recommendations["budget"] = "NVIDIA H100"

    else:  # low budget
        recommendations["budget"] = "NVIDIA L40S"
        recommendations["optimal"] = "NVIDIA A100"

    return {k: v for k, v in recommendations.items() if v is not None}

def estimate_bandwidth_utilization(sequential_ratio: float, burst_size_bytes: int = 64) -> float:
    """Estimate bandwidth utilization based on access patterns.

    Args:
        sequential_ratio: Ratio of sequential accesses (0.0-1.0)
        burst_size_bytes: Average burst size in bytes

    Returns:
        float: Bandwidth utilization factor (0.0-1.0)
    """
    # Base utilization depends on sequential access ratio
    base_utilization = 0.3 + (sequential_ratio * 0.5)  # 30-80% range

    # Larger bursts improve utilization
    burst_factor = min(1.0, burst_size_bytes / 256.0)  # Normalize to 256B

    # Combined utilization
    utilization = base_utilization * (0.8 + 0.2 * burst_factor)

    return min(1.0, utilization)

def calculate_achieved_bandwidth(gpu_type: str, sequential_ratio: float,
                               burst_size_bytes: int = 64) -> float:
    """Calculate achieved bandwidth for given access pattern.

    Args:
        gpu_type: GPU type name
        sequential_ratio: Ratio of sequential accesses (0.0-1.0)
        burst_size_bytes: Average burst size in bytes

    Returns:
        float: Achieved bandwidth in GB/s
    """
    effective_bandwidth = get_gpu_effective_bandwidth(gpu_type)
    utilization = estimate_bandwidth_utilization(sequential_ratio, burst_size_bytes)

    return effective_bandwidth * utilization