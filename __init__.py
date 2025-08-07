"""
Memory Analyzer Package
Simplified memory performance analysis for ML workloads
Enhanced with multi-GPU cluster support
"""

__version__ = "2.1.0"
__author__ = "Joseph Yared"

# Core classes
from .training_analyzer import MemoryAnalyzer, AnalysisResult
from .training_workload import (
    create_workload_model, estimate_max_batch_size,
    MemoryBreakdown, AccessPatterns
)
from .gpu_specs import get_gpu_spec, get_gpu_memory_capacity
from .training_recommendations import RecommendationEngine, get_optimization_recommendations
from .utils import AnalysisType, AnalysisStage, calculate_memory_efficiency

# Multi-GPU support
from .cluster_specs import ClusterSpec, create_cluster_spec, InterconnectType
from .training_parallel import (
    DistributionStrategy, DistributedMemoryAnalyzer, ParallelStrategy,
    create_distribution_strategy
)

# Main API functions
def analyze_memory(model_type: str, model_size_billions: float, batch_size: int,
                  sequence_length: int = 2048, gpu_type: str = "NVIDIA H100",
                  framework: str = "pytorch", enable_recommendations: bool = True,
                  num_gpus: int = 1, distribution_strategy: str = None):
    """Quick memory analysis function with multi-GPU support.

    Args:
        model_type: Model architecture type
        model_size_billions: Model size in billions of parameters
        batch_size: Training batch size
        sequence_length: Sequence length
        gpu_type: GPU type name
        framework: ML framework
        enable_recommendations: Whether to generate recommendations
        num_gpus: Number of GPUs (1 for single-GPU analysis)
        distribution_strategy: Distribution strategy for multi-GPU ("data_parallel", "tensor_parallel", etc.)

    Returns:
        AnalysisResult: Analysis results with recommendations
    """
    # Create cluster specification
    if num_gpus > 1:
        cluster_spec = create_cluster_spec(num_gpus, gpu_type)

        # Create distribution strategy if specified
        dist_strategy = None
        if distribution_strategy:
            dist_strategy = create_distribution_strategy(distribution_strategy, num_gpus)
    else:
        cluster_spec = None
        dist_strategy = None

    # Initialize analyzer
    analyzer = MemoryAnalyzer(
        framework=framework,
        gpu_type=gpu_type,
        cluster_spec=cluster_spec
    )

    analysis_type = AnalysisType.BOTTLENECK if enable_recommendations else AnalysisType.PROGRESSIVE

    return analyzer.analyze_model(
        model_type=model_type,
        model_size_billions=model_size_billions,
        batch_size=batch_size,
        sequence_length=sequence_length,
        analysis_type=analysis_type,
        distribution_strategy=dist_strategy
    )

def find_max_batch_size(model_type: str, model_size_billions: float,
                       sequence_length: int = 2048, gpu_type: str = "NVIDIA H100",
                       num_gpus: int = 1, distribution_strategy: str = None):
    """Find maximum batch size that fits in GPU memory.

    Args:
        model_type: Model architecture type
        model_size_billions: Model size in billions of parameters
        sequence_length: Sequence length
        gpu_type: GPU type name
        num_gpus: Number of GPUs
        distribution_strategy: Distribution strategy for multi-GPU

    Returns:
        int: Maximum batch size
    """
    if num_gpus > 1:
        cluster_spec = create_cluster_spec(num_gpus, gpu_type)
        dist_strategy = create_distribution_strategy(distribution_strategy or "data_parallel", num_gpus)

        analyzer = MemoryAnalyzer(gpu_type=gpu_type, cluster_spec=cluster_spec)
        return analyzer.get_max_batch_size(
            model_type, model_size_billions, sequence_length,
            distribution_strategy=dist_strategy
        )
    else:
        return estimate_max_batch_size(model_type, model_size_billions, sequence_length, gpu_type)

def get_memory_breakdown(model_type: str, model_size_billions: float, batch_size: int,
                        sequence_length: int = 2048, gpu_type: str = "NVIDIA H100",
                        num_gpus: int = 1, distribution_strategy: str = None):
    """Get detailed memory breakdown with multi-GPU support.

    Args:
        model_type: Model architecture type
        model_size_billions: Model size in billions of parameters
        batch_size: Training batch size
        sequence_length: Sequence length
        gpu_type: GPU type name
        num_gpus: Number of GPUs
        distribution_strategy: Distribution strategy for multi-GPU

    Returns:
        MemoryBreakdown or Dict: Memory usage breakdown (single or distributed)
    """
    if num_gpus > 1:
        # Multi-GPU memory breakdown
        cluster_spec = create_cluster_spec(num_gpus, gpu_type)
        dist_strategy = create_distribution_strategy(distribution_strategy or "data_parallel", num_gpus)

        analyzer = MemoryAnalyzer(gpu_type=gpu_type, cluster_spec=cluster_spec)
        result = analyzer.analyze_model(
            model_type, model_size_billions, batch_size, sequence_length,
            distribution_strategy=dist_strategy
        )

        return result.distributed_memory if result.distributed_memory else result.memory_breakdown
    else:
        # Single GPU memory breakdown
        workload_model = create_workload_model(model_type, model_size_billions,
                                             batch_size, sequence_length, gpu_type)
        return workload_model.calculate_memory_breakdown()

def analyze_distributed_strategies(model_type: str, model_size_billions: float,
                                 batch_size: int, num_gpus: int,
                                 gpu_type: str = "NVIDIA H100",
                                 sequence_length: int = 2048):
    """Analyze and recommend optimal distribution strategies for multi-GPU training.

    Args:
        model_type: Model architecture type
        model_size_billions: Model size in billions of parameters
        batch_size: Training batch size
        num_gpus: Number of GPUs
        gpu_type: GPU type name
        sequence_length: Sequence length

    Returns:
        Dict: Strategy analysis and recommendations
    """
    if num_gpus <= 1:
        return {"error": "Multi-GPU analysis requires num_gpus > 1"}

    cluster_spec = create_cluster_spec(num_gpus, gpu_type)
    analyzer = MemoryAnalyzer(gpu_type=gpu_type, cluster_spec=cluster_spec)

    return analyzer.analyze_distributed_strategies(
        model_type, model_size_billions, batch_size, sequence_length
    )

def create_multi_gpu_analyzer(num_gpus: int, gpu_type: str = "NVIDIA H100",
                            interconnect: str = None, framework: str = "pytorch"):
    """Create a multi-GPU memory analyzer.

    Args:
        num_gpus: Number of GPUs in cluster
        gpu_type: GPU type name
        interconnect: Interconnect type (auto-selected if None)
        framework: ML framework

    Returns:
        MemoryAnalyzer: Configured multi-GPU analyzer
    """
    cluster_spec = create_cluster_spec(num_gpus, gpu_type, interconnect)
    return MemoryAnalyzer(framework=framework, gpu_type=gpu_type, cluster_spec=cluster_spec)

def compare_gpu_clusters(model_type: str, model_size_billions: float, batch_size: int,
                        cluster_configs: list, sequence_length: int = 2048):
    """Compare memory requirements across different GPU cluster configurations.

    Args:
        model_type: Model architecture type
        model_size_billions: Model size in billions of parameters
        batch_size: Training batch size
        cluster_configs: List of (num_gpus, gpu_type) tuples
        sequence_length: Sequence length

    Returns:
        Dict: Comparison results across cluster configurations
    """
    results = {}

    for i, (num_gpus, gpu_type) in enumerate(cluster_configs):
        config_name = f"{num_gpus}x_{gpu_type.replace(' ', '_')}"

        try:
            if num_gpus == 1:
                # Single GPU analysis
                result = analyze_memory(
                    model_type, model_size_billions, batch_size,
                    sequence_length, gpu_type, num_gpus=1
                )
                memory_per_gpu = result.memory_breakdown.total_memory_gb
                total_memory = memory_per_gpu
                feasible = memory_per_gpu <= get_gpu_memory_capacity(gpu_type) * 0.9
                strategy = "single_gpu"
            else:
                # Multi-GPU analysis with tensor parallelism (good for memory reduction)
                result = analyze_memory(
                    model_type, model_size_billions, batch_size,
                    sequence_length, gpu_type, num_gpus=num_gpus,
                    distribution_strategy="tensor_parallel"
                )

                if result.distributed_memory:
                    memory_per_gpu = result.distributed_memory["memory_per_gpu"].total_memory_gb
                    total_memory = result.distributed_memory["total_cluster_memory"]
                    feasible = memory_per_gpu <= get_gpu_memory_capacity(gpu_type) * 0.9
                    strategy = "tensor_parallel"
                else:
                    memory_per_gpu = result.memory_breakdown.total_memory_gb
                    total_memory = memory_per_gpu * num_gpus
                    feasible = False
                    strategy = "failed"

            results[config_name] = {
                "num_gpus": num_gpus,
                "gpu_type": gpu_type,
                "strategy": strategy,
                "memory_per_gpu_gb": memory_per_gpu,
                "total_cluster_memory_gb": total_memory,
                "feasible": feasible,
                "gpu_memory_capacity_gb": get_gpu_memory_capacity(gpu_type)
            }

        except Exception as e:
            results[config_name] = {
                "num_gpus": num_gpus,
                "gpu_type": gpu_type,
                "error": str(e),
                "feasible": False
            }

    return results

# Export all public classes and functions
__all__ = [
    # Core classes
    "MemoryAnalyzer",
    "AnalysisResult",
    "RecommendationEngine",
    "MemoryBreakdown",
    "AccessPatterns",

    # Multi-GPU classes
    "ClusterSpec",
    "DistributionStrategy",
    "DistributedMemoryAnalyzer",
    "ParallelStrategy",
    "InterconnectType",

    # Utility functions
    "analyze_memory",
    "find_max_batch_size",
    "get_memory_breakdown",
    "get_optimization_recommendations",
    "create_workload_model",
    "estimate_max_batch_size",
    "get_gpu_spec",
    "get_gpu_memory_capacity",
    "calculate_memory_efficiency",

    # Multi-GPU functions
    "analyze_distributed_strategies",
    "create_multi_gpu_analyzer",
    "compare_gpu_clusters",
    "create_cluster_spec",
    "create_distribution_strategy",

    # Enums
    "AnalysisType",
    "AnalysisStage"
]
