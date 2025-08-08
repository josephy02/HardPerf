import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from enum import Enum

from training_workload import MemoryBreakdown
from cluster_specs import ClusterSpec

logger = logging.getLogger(__name__)

class ParallelStrategy(Enum):
    """Types of parallelization strategies."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"

@dataclass
class DistributionStrategy:
    """Multi-GPU distribution strategy configuration."""
    data_parallel: int = 1
    model_parallel: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1

    def __post_init__(self):
        """Validate strategy configuration."""
        if self.total_gpus <= 0:
            raise ValueError("Total GPUs must be positive")

        # Validate that only one primary strategy is used (for simplicity)
        active_strategies = sum([
            self.data_parallel > 1,
            self.model_parallel > 1,
            self.tensor_parallel > 1,
            self.pipeline_parallel > 1
        ])

        if active_strategies > 1:
            logger.warning("Hybrid parallelism detected. Analysis may be less accurate.")

    @property
    def total_gpus(self) -> int:
        """Calculate total number of GPUs required."""
        return self.data_parallel * self.model_parallel * self.tensor_parallel * self.pipeline_parallel

    @property
    def primary_strategy(self) -> ParallelStrategy:
        """Identify the primary parallelization strategy."""
        active_strategies = sum([
            self.data_parallel > 1,
            self.model_parallel > 1,
            self.tensor_parallel > 1,
            self.pipeline_parallel > 1
        ])

        # If multiple strategies are active, it's hybrid
        if active_strategies > 1:
            return ParallelStrategy.HYBRID
        elif self.tensor_parallel > 1:
            return ParallelStrategy.TENSOR_PARALLEL
        elif self.pipeline_parallel > 1:
            return ParallelStrategy.PIPELINE_PARALLEL
        elif self.model_parallel > 1:
            return ParallelStrategy.MODEL_PARALLEL
        elif self.data_parallel > 1:
            return ParallelStrategy.DATA_PARALLEL
        else:
            return ParallelStrategy.DATA_PARALLEL

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "data_parallel": self.data_parallel,
            "model_parallel": self.model_parallel,
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "total_gpus": self.total_gpus,
            "primary_strategy": self.primary_strategy.value
        }

@dataclass
class CommunicationPattern:
    """Communication pattern characteristics for a distribution strategy."""
    frequency: str  # "per_layer", "per_batch", "per_step"
    data_size_gb: float
    all_reduce_count: int
    point_to_point_count: int

    def estimate_communication_time(self, bandwidth_gb_s: float, latency_us: float) -> float:
        """Estimate communication time in milliseconds."""
        # Data transfer time
        transfer_time_ms = (self.data_size_gb / bandwidth_gb_s) * 1000

        # Latency overhead (convert microseconds to milliseconds)
        latency_overhead_ms = (self.all_reduce_count + self.point_to_point_count) * latency_us / 1000

        # Cap the total time to reasonable bounds
        total_time_ms = transfer_time_ms + latency_overhead_ms
        return min(total_time_ms, 1000.0)  # Cap at 1 second

class DistributedMemoryAnalyzer:
    """Analyzer for multi-GPU memory requirements and performance."""

    def __init__(self, cluster_spec: ClusterSpec):
        """Initialize distributed memory analyzer.

        Args:
            cluster_spec: Cluster configuration
        """
        self.cluster_spec = cluster_spec

    def calculate_distributed_memory(self, base_memory: MemoryBreakdown,
                                   strategy: DistributionStrategy) -> Dict:
        """Calculate memory requirements per GPU for distributed training.

        Args:
            base_memory: Single-GPU memory breakdown
            strategy: Distribution strategy

        Returns:
            Dict: Memory analysis results
        """
        primary_strategy = strategy.primary_strategy

        if primary_strategy == ParallelStrategy.DATA_PARALLEL:
            return self._data_parallel_memory(base_memory, strategy)
        elif primary_strategy == ParallelStrategy.TENSOR_PARALLEL:
            return self._tensor_parallel_memory(base_memory, strategy)
        elif primary_strategy == ParallelStrategy.PIPELINE_PARALLEL:
            return self._pipeline_parallel_memory(base_memory, strategy)
        elif primary_strategy == ParallelStrategy.MODEL_PARALLEL:
            return self._model_parallel_memory(base_memory, strategy)
        else:  # HYBRID
            return self._hybrid_parallel_memory(base_memory, strategy)

    def _data_parallel_memory(self, base_memory: MemoryBreakdown,
                             strategy: DistributionStrategy) -> Dict:
        """Calculate memory for data parallelism."""
        # FIXED: For data parallelism, recalculate with per-GPU batch size
        from training_workload import create_workload_model

        # Calculate per-GPU batch size
        per_gpu_batch_size = base_memory.total_memory_gb / base_memory.model_parameters_gb  # Rough estimate
        effective_batch_size = max(1, int(8 / strategy.data_parallel))  # Assume base batch 8

        # For data parallel, model parameters stay the same per GPU
        per_gpu_memory = MemoryBreakdown(
            model_parameters_gb=base_memory.model_parameters_gb,
            optimizer_states_gb=base_memory.optimizer_states_gb,
            gradients_gb=base_memory.gradients_gb,
            activations_gb=base_memory.activations_gb / strategy.data_parallel,  # Batch split
            temporary_buffers_gb=base_memory.temporary_buffers_gb / strategy.data_parallel,  # Also scales with batch
            framework_overhead_gb=base_memory.framework_overhead_gb,
            fragmentation_gb=base_memory.fragmentation_gb * 0.9,  # Slightly less fragmentation
            total_memory_gb=0.0  # Will be calculated
        )

        # Add gradient synchronization buffers (small for data parallel)
        gradient_sync_buffer = base_memory.gradients_gb * 0.05  # 5% buffer for async
        per_gpu_memory.temporary_buffers_gb += gradient_sync_buffer

        # Recalculate total
        per_gpu_memory.total_memory_gb = (
            per_gpu_memory.model_parameters_gb + per_gpu_memory.optimizer_states_gb +
            per_gpu_memory.gradients_gb + per_gpu_memory.activations_gb +
            per_gpu_memory.temporary_buffers_gb + per_gpu_memory.framework_overhead_gb +
            per_gpu_memory.fragmentation_gb
        )

        # Communication pattern (low overhead for data parallel)
        comm_pattern = CommunicationPattern(
            frequency="per_batch",
            data_size_gb=base_memory.gradients_gb,
            all_reduce_count=1,
            point_to_point_count=0
        )

        return {
            "memory_per_gpu": per_gpu_memory,
            "total_cluster_memory": per_gpu_memory.total_memory_gb * strategy.total_gpus,
            "communication_pattern": comm_pattern,
            "memory_efficiency": base_memory.model_parameters_gb / per_gpu_memory.total_memory_gb,
            "parallelization_efficiency": 0.95  # Data parallel has good efficiency
        }

    def _tensor_parallel_memory(self, base_memory: MemoryBreakdown,
                               strategy: DistributionStrategy) -> Dict:
        """Calculate memory for tensor parallelism."""
        tp_degree = strategy.tensor_parallel

        # All components scale down with tensor parallelism
        per_gpu_memory = MemoryBreakdown(
            model_parameters_gb=base_memory.model_parameters_gb / tp_degree,
            optimizer_states_gb=base_memory.optimizer_states_gb / tp_degree,
            gradients_gb=base_memory.gradients_gb / tp_degree,
            activations_gb=base_memory.activations_gb * 0.8,  # Some reduction due to sharding
            temporary_buffers_gb=base_memory.temporary_buffers_gb / tp_degree,
            framework_overhead_gb=base_memory.framework_overhead_gb,  # Doesn't scale
            fragmentation_gb=base_memory.fragmentation_gb / tp_degree,
            total_memory_gb=0.0
        )

        # Add all-reduce communication buffers
        allreduce_buffer = base_memory.activations_gb * 0.15  # 15% buffer for all-reduce
        per_gpu_memory.temporary_buffers_gb += allreduce_buffer

        # Recalculate total
        per_gpu_memory.total_memory_gb = (
            per_gpu_memory.model_parameters_gb + per_gpu_memory.optimizer_states_gb +
            per_gpu_memory.gradients_gb + per_gpu_memory.activations_gb +
            per_gpu_memory.temporary_buffers_gb + per_gpu_memory.framework_overhead_gb +
            per_gpu_memory.fragmentation_gb
        )

        # Communication pattern (frequent all-reduce operations)
        comm_pattern = CommunicationPattern(
            frequency="per_layer",
            data_size_gb=base_memory.activations_gb * 0.5,  # Partial activations
            all_reduce_count=4,  # Multiple all-reduces per layer
            point_to_point_count=0
        )

        return {
            "memory_per_gpu": per_gpu_memory,
            "total_cluster_memory": per_gpu_memory.total_memory_gb * strategy.total_gpus,
            "communication_pattern": comm_pattern,
            "memory_efficiency": base_memory.model_parameters_gb / (per_gpu_memory.total_memory_gb * tp_degree),
            "parallelization_efficiency": 0.7  # Communication overhead reduces efficiency
        }

    def _pipeline_parallel_memory(self, base_memory: MemoryBreakdown,
                                 strategy: DistributionStrategy) -> Dict:
        """Calculate memory for pipeline parallelism."""
        pp_degree = strategy.pipeline_parallel

        # Model parameters and optimizer states scale down
        per_gpu_memory = MemoryBreakdown(
            model_parameters_gb=base_memory.model_parameters_gb / pp_degree,
            optimizer_states_gb=base_memory.optimizer_states_gb / pp_degree,
            gradients_gb=base_memory.gradients_gb / pp_degree,
            activations_gb=base_memory.activations_gb,  # Need to store activations for pipeline
            temporary_buffers_gb=base_memory.temporary_buffers_gb,
            framework_overhead_gb=base_memory.framework_overhead_gb,
            fragmentation_gb=base_memory.fragmentation_gb / pp_degree,
            total_memory_gb=0.0
        )

        # Add pipeline buffers (need to store multiple micro-batches)
        pipeline_buffers = base_memory.activations_gb * 0.3  # 30% for pipeline buffers
        per_gpu_memory.temporary_buffers_gb += pipeline_buffers

        # Recalculate total
        per_gpu_memory.total_memory_gb = (
            per_gpu_memory.model_parameters_gb + per_gpu_memory.optimizer_states_gb +
            per_gpu_memory.gradients_gb + per_gpu_memory.activations_gb +
            per_gpu_memory.temporary_buffers_gb + per_gpu_memory.framework_overhead_gb +
            per_gpu_memory.fragmentation_gb
        )

        # Communication pattern (point-to-point between pipeline stages)
        comm_pattern = CommunicationPattern(
            frequency="per_step",
            data_size_gb=base_memory.activations_gb / pp_degree,
            all_reduce_count=0,
            point_to_point_count=pp_degree - 1  # Between adjacent stages
        )

        return {
            "memory_per_gpu": per_gpu_memory,
            "total_cluster_memory": per_gpu_memory.total_memory_gb * strategy.total_gpus,
            "communication_pattern": comm_pattern,
            "memory_efficiency": base_memory.model_parameters_gb / (per_gpu_memory.total_memory_gb * pp_degree),
            "parallelization_efficiency": 0.6  # Pipeline bubbles reduce efficiency
        }

    def _model_parallel_memory(self, base_memory: MemoryBreakdown,
                              strategy: DistributionStrategy) -> Dict:
        """Calculate memory for model parallelism (layer-wise)."""
        mp_degree = strategy.model_parallel

        # Similar to pipeline but without pipeline-specific optimizations
        per_gpu_memory = MemoryBreakdown(
            model_parameters_gb=base_memory.model_parameters_gb / mp_degree,
            optimizer_states_gb=base_memory.optimizer_states_gb / mp_degree,
            gradients_gb=base_memory.gradients_gb / mp_degree,
            activations_gb=base_memory.activations_gb * 0.9,  # Slight reduction
            temporary_buffers_gb=base_memory.temporary_buffers_gb,
            framework_overhead_gb=base_memory.framework_overhead_gb,
            fragmentation_gb=base_memory.fragmentation_gb / mp_degree,
            total_memory_gb=0.0
        )

        # Recalculate total
        per_gpu_memory.total_memory_gb = (
            per_gpu_memory.model_parameters_gb + per_gpu_memory.optimizer_states_gb +
            per_gpu_memory.gradients_gb + per_gpu_memory.activations_gb +
            per_gpu_memory.temporary_buffers_gb + per_gpu_memory.framework_overhead_gb +
            per_gpu_memory.fragmentation_gb
        )

        # Communication pattern
        comm_pattern = CommunicationPattern(
            frequency="per_layer",
            data_size_gb=base_memory.activations_gb / mp_degree,
            all_reduce_count=0,
            point_to_point_count=mp_degree - 1
        )

        return {
            "memory_per_gpu": per_gpu_memory,
            "total_cluster_memory": per_gpu_memory.total_memory_gb * strategy.total_gpus,
            "communication_pattern": comm_pattern,
            "memory_efficiency": base_memory.model_parameters_gb / (per_gpu_memory.total_memory_gb * mp_degree),
            "parallelization_efficiency": 0.5  # Sequential dependencies hurt efficiency
        }

    def _hybrid_parallel_memory(self, base_memory: MemoryBreakdown,
                               strategy: DistributionStrategy) -> Dict:
        """Calculate memory for hybrid parallelism - NEW METHOD."""
        # Hybrid parallelism combines multiple strategies
        # Apply reductions in order: tensor -> pipeline -> data

        # Start with base memory
        current_memory = MemoryBreakdown(
            model_parameters_gb=base_memory.model_parameters_gb,
            optimizer_states_gb=base_memory.optimizer_states_gb,
            gradients_gb=base_memory.gradients_gb,
            activations_gb=base_memory.activations_gb,
            temporary_buffers_gb=base_memory.temporary_buffers_gb,
            framework_overhead_gb=base_memory.framework_overhead_gb,
            fragmentation_gb=base_memory.fragmentation_gb,
            total_memory_gb=base_memory.total_memory_gb
        )

        # Apply tensor parallelism first (model parameters scale down)
        if strategy.tensor_parallel > 1:
            tp_degree = strategy.tensor_parallel
            current_memory.model_parameters_gb /= tp_degree
            current_memory.optimizer_states_gb /= tp_degree
            current_memory.gradients_gb /= tp_degree
            current_memory.temporary_buffers_gb /= tp_degree
            current_memory.fragmentation_gb /= tp_degree
            # Add tensor parallel communication buffers
            current_memory.temporary_buffers_gb += base_memory.activations_gb * 0.15

        # Apply pipeline parallelism (model components scale further)
        if strategy.pipeline_parallel > 1:
            pp_degree = strategy.pipeline_parallel
            current_memory.model_parameters_gb /= pp_degree
            current_memory.optimizer_states_gb /= pp_degree
            current_memory.gradients_gb /= pp_degree
            current_memory.fragmentation_gb /= pp_degree
            # Add pipeline buffers
            current_memory.temporary_buffers_gb += base_memory.activations_gb * 0.3

        # Apply data parallelism last (activations scale down with batch size)
        if strategy.data_parallel > 1:
            dp_degree = strategy.data_parallel
            current_memory.activations_gb /= dp_degree
            current_memory.temporary_buffers_gb /= dp_degree
            # Add gradient synchronization buffers
            current_memory.temporary_buffers_gb += base_memory.gradients_gb * 0.05

        # Recalculate total
        current_memory.total_memory_gb = (
            current_memory.model_parameters_gb + current_memory.optimizer_states_gb +
            current_memory.gradients_gb + current_memory.activations_gb +
            current_memory.temporary_buffers_gb + current_memory.framework_overhead_gb +
            current_memory.fragmentation_gb
        )

        # Complex communication pattern for hybrid
        comm_pattern = CommunicationPattern(
            frequency="per_layer",  # Most restrictive
            data_size_gb=base_memory.activations_gb * 0.3,  # Mixed traffic
            all_reduce_count=2,  # Both tensor and data parallel
            point_to_point_count=max(0, strategy.pipeline_parallel - 1)  # Pipeline stages
        )

        # Lower efficiency due to complexity
        parallelization_efficiency = 0.6 * (0.9 ** (sum([
            strategy.tensor_parallel > 1,
            strategy.pipeline_parallel > 1,
            strategy.data_parallel > 1
        ]) - 1))  # Penalty for each additional strategy

        return {
            "memory_per_gpu": current_memory,
            "total_cluster_memory": current_memory.total_memory_gb * strategy.total_gpus,
            "communication_pattern": comm_pattern,
            "memory_efficiency": base_memory.model_parameters_gb / (current_memory.total_memory_gb * strategy.total_gpus),
            "parallelization_efficiency": parallelization_efficiency
        }

    def analyze_communication_overhead(self, strategy: DistributionStrategy,
                                     memory_analysis: Dict) -> Dict:
        """Analyze communication overhead for the given strategy."""
        comm_pattern = memory_analysis["communication_pattern"]

        # Get cluster bandwidth and latency
        bandwidth_gb_s = self.cluster_spec.get_effective_bandwidth()
        latency_us = self.cluster_spec.estimate_communication_latency()

        # Calculate communication time
        comm_time_ms = comm_pattern.estimate_communication_time(bandwidth_gb_s, latency_us)

        # Estimate compute time (rough approximation)
        # Assume 1 TFLOP/s effective compute, model size determines FLOPs
        model_params = memory_analysis["memory_per_gpu"].model_parameters_gb * 1e9  # Convert to parameters
        flops_per_token = model_params * 6  # 6 FLOPs per parameter per token (forward + backward)

        # Assume 1000 TFLOP/s effective compute rate for modern GPUs
        compute_time_ms = flops_per_token / (1000e12) * 1000  # Convert to milliseconds

        # Calculate communication to compute ratio
        comm_compute_ratio = comm_time_ms / max(compute_time_ms, 1e-6)

        # Cap communication overhead at reasonable bounds
        comm_overhead_percent = min(comm_compute_ratio / (1 + comm_compute_ratio) * 100, 95.0)

        return {
            "communication_time_ms": comm_time_ms,
            "estimated_compute_time_ms": compute_time_ms,
            "communication_compute_ratio": comm_compute_ratio,
            "communication_overhead_percent": comm_overhead_percent,
            "effective_bandwidth_gb_s": bandwidth_gb_s,
            "estimated_latency_us": latency_us,
            "scalability_factor": memory_analysis["parallelization_efficiency"]
        }

    def recommend_optimal_strategy(self, base_memory: MemoryBreakdown,
                                  target_batch_size: int = 32) -> Dict:
        """Recommend optimal distribution strategy for given constraints.

        Args:
            base_memory: Single-GPU memory breakdown
            target_batch_size: Desired batch size

        Returns:
            Dict: Strategy recommendations with analysis
        """
        gpu_memory_gb = self.cluster_spec.gpu_spec.memory_capacity_gb
        available_memory_gb = gpu_memory_gb * 0.9  # 90% utilization target

        strategies_to_test = []

        # Test data parallelism
        if base_memory.total_memory_gb <= available_memory_gb:
            dp_degree = min(self.cluster_spec.num_gpus, target_batch_size)
            strategies_to_test.append(DistributionStrategy(data_parallel=dp_degree))

        # Test tensor parallelism
        for tp_degree in [2, 4, 8]:
            if tp_degree <= self.cluster_spec.num_gpus:
                strategies_to_test.append(DistributionStrategy(tensor_parallel=tp_degree))

        # Test pipeline parallelism
        for pp_degree in [2, 4, 8]:
            if pp_degree <= self.cluster_spec.num_gpus:
                strategies_to_test.append(DistributionStrategy(pipeline_parallel=pp_degree))

        # Analyze each strategy
        results = {}
        best_strategy = None
        best_score = -1

        for i, strategy in enumerate(strategies_to_test):
            try:
                memory_analysis = self.calculate_distributed_memory(base_memory, strategy)
                comm_analysis = self.analyze_communication_overhead(strategy, memory_analysis)

                per_gpu_memory = memory_analysis["memory_per_gpu"].total_memory_gb

                # Check if strategy fits in memory
                if per_gpu_memory > available_memory_gb:
                    feasible = False
                    score = 0
                else:
                    feasible = True
                    # Score based on memory efficiency and communication overhead
                    memory_efficiency = memory_analysis["memory_efficiency"]
                    comm_overhead = comm_analysis["communication_overhead_percent"]
                    parallelization_eff = memory_analysis["parallelization_efficiency"]

                    score = (memory_efficiency * 0.4 +
                            (100 - comm_overhead) / 100 * 0.4 +
                            parallelization_eff * 0.2)

                results[f"strategy_{i+1}"] = {
                    "strategy": strategy.to_dict(),
                    "memory_per_gpu_gb": per_gpu_memory,
                    "total_cluster_memory_gb": memory_analysis["total_cluster_memory"],
                    "communication_overhead_percent": comm_analysis["communication_overhead_percent"],
                    "memory_efficiency": memory_analysis["memory_efficiency"],
                    "parallelization_efficiency": memory_analysis["parallelization_efficiency"],
                    "feasible": feasible,
                    "score": score
                }

                if feasible and score > best_score:
                    best_score = score
                    best_strategy = f"strategy_{i+1}"

            except Exception as e:
                logger.warning(f"Failed to analyze strategy {strategy.to_dict()}: {e}")

        return {
            "recommended_strategy": best_strategy,
            "all_strategies": results,
            "cluster_info": self.cluster_spec.get_topology_info()
        }

def create_distribution_strategy(strategy_type: str, num_gpus: int,
                                data_parallel: int = None, tensor_parallel: int = None,
                                pipeline_parallel: int = None, model_parallel: int = None) -> DistributionStrategy:
    """Factory function to create distribution strategy with support for hybrid configurations.

    Args:
        strategy_type: Type of strategy ("data_parallel", "tensor_parallel", "hybrid", etc.)
        num_gpus: Number of GPUs to use
        data_parallel: Data parallel degree (for hybrid)
        tensor_parallel: Tensor parallel degree (for hybrid)
        pipeline_parallel: Pipeline parallel degree (for hybrid)
        model_parallel: Model parallel degree (for hybrid)

    Returns:
        DistributionStrategy: Configured strategy
    """
    if strategy_type == "data_parallel":
        return DistributionStrategy(data_parallel=num_gpus)
    elif strategy_type == "tensor_parallel":
        return DistributionStrategy(tensor_parallel=num_gpus)
    elif strategy_type == "pipeline_parallel":
        return DistributionStrategy(pipeline_parallel=num_gpus)
    elif strategy_type == "model_parallel":
        return DistributionStrategy(model_parallel=num_gpus)
    elif strategy_type == "hybrid":
        # For hybrid, use provided degrees or distribute evenly
        if data_parallel is None and tensor_parallel is None and pipeline_parallel is None:
            # If no specific degrees provided, create a balanced hybrid
            if num_gpus == 64:  # Example from your YAML
                return DistributionStrategy(
                    data_parallel=4,
                    tensor_parallel=4,
                    pipeline_parallel=4
                )
            elif num_gpus >= 16:
                # Try to split into 3 dimensions
                cube_root = round(num_gpus ** (1/3))
                return DistributionStrategy(
                    data_parallel=cube_root,
                    tensor_parallel=cube_root,
                    pipeline_parallel=num_gpus // (cube_root * cube_root)
                )
            elif num_gpus >= 8:
                # Split into 2 dimensions
                sqrt_gpus = int(num_gpus ** 0.5)
                return DistributionStrategy(
                    data_parallel=sqrt_gpus,
                    tensor_parallel=num_gpus // sqrt_gpus
                )
            else:
                # Small clusters: use data parallel
                return DistributionStrategy(data_parallel=num_gpus)
        else:
            # Use provided degrees
            return DistributionStrategy(
                data_parallel=data_parallel or 1,
                tensor_parallel=tensor_parallel or 1,
                pipeline_parallel=pipeline_parallel or 1,
                model_parallel=model_parallel or 1
            )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

def estimate_scaling_efficiency(base_time_seconds: float,
                               strategy: DistributionStrategy,
                               communication_overhead_percent: float) -> Dict:
    """Estimate scaling efficiency for given strategy.

    Args:
        base_time_seconds: Single-GPU training time
        strategy: Distribution strategy
        communication_overhead_percent: Communication overhead percentage

    Returns:
        Dict: Scaling efficiency analysis
    """
    ideal_speedup = strategy.total_gpus

    # Apply communication overhead
    comm_factor = 1 + (communication_overhead_percent / 100)
    actual_speedup = ideal_speedup / comm_factor

    # Apply strategy-specific efficiency factors
    if strategy.primary_strategy == ParallelStrategy.DATA_PARALLEL:
        strategy_efficiency = 0.95  # Data parallel scales well
    elif strategy.primary_strategy == ParallelStrategy.TENSOR_PARALLEL:
        strategy_efficiency = 0.70  # High communication overhead
    elif strategy.primary_strategy == ParallelStrategy.PIPELINE_PARALLEL:
        strategy_efficiency = 0.60  # Pipeline bubbles
    elif strategy.primary_strategy == ParallelStrategy.HYBRID:
        strategy_efficiency = 0.50  # Complex hybrid has overhead
    else:
        strategy_efficiency = 0.50  # Model parallel has dependencies

    actual_speedup *= strategy_efficiency

    estimated_time = base_time_seconds / actual_speedup
    efficiency_percent = (actual_speedup / ideal_speedup) * 100

    return {
        "ideal_speedup": ideal_speedup,
        "actual_speedup": actual_speedup,
        "efficiency_percent": efficiency_percent,
        "estimated_training_time_seconds": estimated_time,
        "estimated_training_time_hours": estimated_time / 3600,
        "communication_overhead_percent": communication_overhead_percent,
        "strategy_efficiency_factor": strategy_efficiency
    }


def calculate_optimal_micro_batch_size(strategy: DistributionStrategy,
                                     target_batch_size: int,
                                     sequence_length: int = 2048) -> Dict:
    """Calculate optimal micro-batch size for pipeline parallelism.

    Args:
        strategy: Distribution strategy
        target_batch_size: Global batch size target
        sequence_length: Sequence length

    Returns:
        Dict: Micro-batch size recommendations
    """
    if strategy.primary_strategy != ParallelStrategy.PIPELINE_PARALLEL and strategy.pipeline_parallel <= 1:
        return {
            "micro_batch_size": target_batch_size,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": target_batch_size
        }

    pp_degree = strategy.pipeline_parallel

    # For pipeline parallelism, we need to balance:
    # 1. Memory usage (smaller micro-batches use less memory)
    # 2. Pipeline efficiency (more micro-batches reduce bubble overhead)

    # Rule of thumb: use 4-8 micro-batches per pipeline stage
    optimal_micro_batches = pp_degree * 4
    micro_batch_size = max(1, target_batch_size // optimal_micro_batches)

    # Calculate actual configuration
    if micro_batch_size * optimal_micro_batches < target_batch_size:
        gradient_accumulation_steps = math.ceil(target_batch_size / (micro_batch_size * optimal_micro_batches))
    else:
        gradient_accumulation_steps = 1

    effective_batch_size = micro_batch_size * optimal_micro_batches * gradient_accumulation_steps

    # Estimate pipeline efficiency
    bubble_overhead = 1.0 / optimal_micro_batches  # Rough approximation
    pipeline_efficiency = 1.0 - bubble_overhead

    return {
        "micro_batch_size": micro_batch_size,
        "num_micro_batches": optimal_micro_batches,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "pipeline_efficiency": pipeline_efficiency,
        "estimated_bubble_overhead_percent": bubble_overhead * 100
    }


def get_strategy_recommendations_summary(recommendations: Dict) -> str:
    """Generate human-readable summary of strategy recommendations.

    Args:
        recommendations: Output from recommend_optimal_strategy

    Returns:
        str: Formatted summary
    """
    if not recommendations.get("recommended_strategy"):
        return "No feasible strategy found for the given constraints."

    best = recommendations["recommended_strategy"]
    best_config = recommendations["all_strategies"][best]

    strategy_config = best_config["strategy"]
    primary_strategy = strategy_config.get("primary_strategy", "unknown")

    summary_lines = [
        f"RECOMMENDED STRATEGY: {primary_strategy.replace('_', ' ').title()}",
        f"GPUs Required: {strategy_config.get('total_gpus', 'unknown')}",
        f"Memory per GPU: {best_config['memory_per_gpu_gb']:.1f} GB",
        f"Total Cluster Memory: {best_config['total_cluster_memory_gb']:.1f} GB",
        f"Communication Overhead: {best_config['communication_overhead_percent']:.1f}%",
        f"Memory Efficiency: {best_config['memory_efficiency']:.1%}",
        f"Parallelization Efficiency: {best_config['parallelization_efficiency']:.1%}",
        ""
    ]

    # Add configuration details
    if primary_strategy == "data_parallel":
        summary_lines.append("✓ Each GPU holds full model copy")
        summary_lines.append("✓ Batch is split across GPUs")
        summary_lines.append("✓ Low communication overhead")
    elif primary_strategy == "tensor_parallel":
        summary_lines.append("✓ Model tensors split across GPUs")
        summary_lines.append("✓ Significant memory savings")
        summary_lines.append("⚠ High communication overhead")
    elif primary_strategy == "pipeline_parallel":
        summary_lines.append("✓ Model layers distributed across GPUs")
        summary_lines.append("✓ Good memory efficiency")
        summary_lines.append("⚠ Pipeline bubble overhead")
    elif primary_strategy == "hybrid":
        summary_lines.append("✓ Combines multiple parallelization strategies")
        summary_lines.append("✓ Maximum memory savings for large models")
        summary_lines.append("⚠ Complex coordination and communication")

    return "\n".join(summary_lines)
