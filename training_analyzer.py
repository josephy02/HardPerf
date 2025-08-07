"""
Author              :
Copyright           :
File Name           : training_analyzer.py
Description         : Memory Performance Analyzer
                      Consolidated memory analysis with progressive pipeline and bottleneck detection


Revision History    :
Date                  Author               Comments
--------------------------------------------------------------------------------------------------

"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from utils import AnalysisType, AnalysisStage, calculate_memory_efficiency
from training_workload import (
    create_workload_model, estimate_max_batch_size, MemoryBreakdown, AccessPatterns
)
from gpu_specs import get_gpu_spec, get_gpu_memory_capacity
from cluster_specs import ClusterSpec, create_cluster_spec
from training_parallel import DistributionStrategy, DistributedMemoryAnalyzer, create_distribution_strategy
from training_estimator import TrainingEstimator, TrainingEstimate

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Complete analysis result container with multi-GPU support and training estimation."""
    analysis_type: AnalysisType
    analysis_stage: AnalysisStage
    model_info: Dict[str, Any]
    memory_breakdown: MemoryBreakdown
    access_patterns: Optional[AccessPatterns] = None
    bottlenecks: Optional[List[Dict]] = None
    recommendations: Optional[List[Dict]] = None
    bandwidth_metrics: Optional[Dict] = None
    execution_time_seconds: float = 0.0

    # Multi-GPU specific fields
    cluster_spec: Optional[ClusterSpec] = None
    distribution_strategy: Optional[DistributionStrategy] = None
    distributed_memory: Optional[Dict] = None
    communication_analysis: Optional[Dict] = None
    scaling_analysis: Optional[Dict] = None

    # Training estimation field
    training_estimate: Optional[TrainingEstimate] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            "analysis_type": self.analysis_type.value,
            "analysis_stage": self.analysis_stage.value,
            "model_info": self.model_info,
            "memory_breakdown": self.memory_breakdown.to_dict(),
            "execution_time_seconds": self.execution_time_seconds
        }

        if self.access_patterns:
            result["access_patterns"] = self.access_patterns.to_dict()

        if self.bottlenecks:
            result["bottlenecks"] = self.bottlenecks

        if self.recommendations:
            result["recommendations"] = self.recommendations

        if self.bandwidth_metrics:
            result["bandwidth_metrics"] = self.bandwidth_metrics

        # Multi-GPU fields
        if self.cluster_spec:
            result["cluster_spec"] = self.cluster_spec.get_topology_info()

        if self.distribution_strategy:
            result["distribution_strategy"] = self.distribution_strategy.to_dict()

        if self.distributed_memory:
            result["distributed_memory"] = self.distributed_memory

        if self.communication_analysis:
            result["communication_analysis"] = self.communication_analysis

        if self.scaling_analysis:
            result["scaling_analysis"] = self.scaling_analysis

        # Training estimation field
        if self.training_estimate:
            result["training_estimate"] = self.training_estimate.to_dict()

        return result

    def get_summary(self) -> str:
        """Get text summary of analysis."""
        model_type = self.model_info.get("model_type", "Unknown")
        model_size = self.model_info.get("model_size_billions", 0)
        batch_size = self.model_info.get("batch_size", 0)

        if self.cluster_spec and self.cluster_spec.num_gpus > 1:
            # Multi-GPU summary
            total_memory = self.distributed_memory["total_cluster_memory"] if self.distributed_memory else 0
            memory_per_gpu = self.distributed_memory["memory_per_gpu"].total_memory_gb if self.distributed_memory else 0

            summary = [
                f"Multi-GPU Memory Analysis: {model_type} ({model_size}B parameters)",
                f"Cluster: {self.cluster_spec.num_gpus}x {self.cluster_spec.gpu_type}",
                f"Strategy: {self.distribution_strategy.primary_strategy.value}" if self.distribution_strategy else "Unknown",
                f"Batch Size: {batch_size}, Memory per GPU: {memory_per_gpu:.2f} GB",
                f"Total Cluster Memory: {total_memory:.2f} GB",
                f"Analysis Stage: {self.analysis_stage.value.title()}"
            ]
        else:
            # Single GPU summary
            total_memory = self.memory_breakdown.total_memory_gb
            summary = [
                f"Memory Analysis: {model_type} ({model_size}B parameters)",
                f"Batch Size: {batch_size}, Total Memory: {total_memory:.2f} GB",
                f"Analysis Stage: {self.analysis_stage.value.title()}"
            ]

        # Add training time summary
        if self.training_estimate:
            days = self.training_estimate.total_hours / 24
            summary.append(f"Training Time: {days:.1f} days ({self.training_estimate.total_hours:.0f} hours)")

        if self.bottlenecks:
            summary.append(f"Bottlenecks Found: {len(self.bottlenecks)}")

        if self.recommendations:
            summary.append(f"Recommendations: {len(self.recommendations)}")

        return "\n".join(summary)

class MemoryAnalyzer:
    """Main memory performance analyzer with multi-GPU support and training estimation."""

    def __init__(self, framework: str = "pytorch", gpu_type: str = "NVIDIA H100",
                 enable_detailed_analysis: bool = True, cluster_spec: Optional[ClusterSpec] = None):
        """Initialize analyzer.

        Args:
            framework: ML framework name
            gpu_type: GPU type for analysis
            enable_detailed_analysis: Enable access pattern and bottleneck analysis
            cluster_spec: Multi-GPU cluster specification (optional)
        """
        self.framework = framework
        self.gpu_type = gpu_type
        self.enable_detailed_analysis = enable_detailed_analysis

        # Set up cluster specification
        if cluster_spec:
            self.cluster_spec = cluster_spec
            # Ensure cluster GPU type matches analyzer GPU type
            if cluster_spec.gpu_type != gpu_type:
                logger.warning(f"Cluster GPU type ({cluster_spec.gpu_type}) differs from analyzer GPU type ({gpu_type})")
                self.gpu_type = cluster_spec.gpu_type
        else:
            self.cluster_spec = create_cluster_spec(1, gpu_type)

        # Validate GPU type
        self.gpu_spec = get_gpu_spec(self.gpu_type)

        # Initialize distributed analyzer if multi-GPU
        if self.cluster_spec.num_gpus > 1:
            self.distributed_analyzer = DistributedMemoryAnalyzer(self.cluster_spec)
            logger.info(f"Initialized multi-GPU analyzer for {self.cluster_spec.num_gpus}x {self.gpu_spec.name}")
        else:
            self.distributed_analyzer = None
            logger.info(f"Initialized single-GPU analyzer for {self.gpu_spec.name}")

        # Initialize training estimator
        self.training_estimator = TrainingEstimator()

    def analyze_model(self, model_type: str, model_size_billions: float,
                     batch_size: int, sequence_length: int = 2048,
                     hidden_dim: Optional[int] = None,
                     analysis_type: AnalysisType = AnalysisType.PROGRESSIVE,
                     distribution_strategy: Optional[DistributionStrategy] = None,
                     **kwargs) -> AnalysisResult:
        """Analyze model memory requirements with optional multi-GPU support and training estimation.

        Args:
            model_type: Model architecture type
            model_size_billions: Model size in billions of parameters
            batch_size: Training batch size
            sequence_length: Sequence length
            hidden_dim: Hidden dimension (estimated if None)
            analysis_type: Type of analysis to perform
            distribution_strategy: Multi-GPU distribution strategy (optional)
            **kwargs: Additional model-specific parameters

        Returns:
            AnalysisResult: Complete analysis results
        """
        start_time = time.time()

        # Create model info
        model_info = {
            "model_type": model_type,
            "model_size_billions": model_size_billions,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "hidden_dim": hidden_dim,
            "framework": self.framework,
            "gpu_type": self.gpu_type,
            "is_multi_gpu": self.cluster_spec.num_gpus > 1
        }

        logger.info(f"Starting {analysis_type.value} analysis for {model_type} "
                   f"({model_size_billions}B params, batch={batch_size})")

        if self.cluster_spec.num_gpus > 1:
            logger.info(f"Multi-GPU cluster: {self.cluster_spec.num_gpus}x {self.gpu_type}")

        try:
            if analysis_type == AnalysisType.PROGRESSIVE:
                result = self._run_progressive_analysis(model_info, distribution_strategy, **kwargs)
            else:  # BOTTLENECK
                result = self._run_bottleneck_analysis(model_info, distribution_strategy, **kwargs)

            result.execution_time_seconds = time.time() - start_time
            logger.info(f"Analysis completed in {result.execution_time_seconds:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def _run_progressive_analysis(self, model_info: Dict,
                                 distribution_strategy: Optional[DistributionStrategy] = None,
                                 **kwargs) -> AnalysisResult:
        """Run progressive 3-stage analysis with optional multi-GPU support and training estimation."""
        # Stage 1: Memory Estimation
        logger.info("Stage 1/3: ... Memory Estimation")

        # Calculate effective batch size for multi-GPU setups
        effective_batch_size = model_info["batch_size"]
        if self.cluster_spec and self.cluster_spec.num_gpus > 1 and distribution_strategy:
            if distribution_strategy.primary_strategy.value == "data_parallel":
                # For data parallelism, batch is split across GPUs
                effective_batch_size = max(1, model_info["batch_size"] / distribution_strategy.data_parallel)
                logger.info(f"Data parallel: effective batch size per GPU = {effective_batch_size}")

        workload_model = create_workload_model(
            model_info["model_type"], model_info["model_size_billions"],
            model_info["batch_size"], model_info["sequence_length"],
            self.gpu_type, effective_batch_size=effective_batch_size, **kwargs
        )

        memory_breakdown = workload_model.calculate_memory_breakdown()

        # Update model_info with estimated hidden_dim
        if model_info["hidden_dim"] is None:
            model_info["hidden_dim"] = workload_model.hidden_dim

        # Stage 2: Access Pattern Analysis (if enabled)
        access_patterns = None
        if self.enable_detailed_analysis:
            logger.info("Stage 2/3: ... Access Pattern Analysis")
            access_patterns = workload_model.analyze_access_patterns()

        # Stage 3: Performance Profiling
        logger.info("Stage 3/3: ... Performance Profiling")
        bandwidth_metrics = self._calculate_bandwidth_metrics(workload_model, access_patterns)

        # Multi-GPU Analysis (if applicable)
        distributed_memory = None
        communication_analysis = None
        scaling_analysis = None

        if self.distributed_analyzer and distribution_strategy:
            logger.info("Stage 3b/3: Multi-GPU Analysis")
            distributed_memory = self.distributed_analyzer.calculate_distributed_memory(
                memory_breakdown, distribution_strategy
            )
            communication_analysis = self.distributed_analyzer.analyze_communication_overhead(
                distribution_strategy, distributed_memory
            )

            # Simple scaling analysis
            ideal_speedup = distribution_strategy.total_gpus
            actual_speedup = ideal_speedup * distributed_memory.get("parallelization_efficiency", 0.8)
            scaling_analysis = {
                "ideal_speedup": ideal_speedup,
                "actual_speedup": actual_speedup,
                "efficiency_percent": (actual_speedup / ideal_speedup) * 100
            }

        # Training Time Estimation
        training_estimate = None
        try:
            logger.info("Stage 3c/3: Training Time Estimation")
            training_estimate = self.training_estimator.estimate_training_time(model_info, self.cluster_spec)
        except Exception as e:
            logger.warning(f"Training estimation failed: {e}")

        return AnalysisResult(
            analysis_type=AnalysisType.PROGRESSIVE,
            analysis_stage=AnalysisStage.PROFILING,
            model_info=model_info,
            memory_breakdown=memory_breakdown,
            access_patterns=access_patterns,
            bandwidth_metrics=bandwidth_metrics,
            cluster_spec=self.cluster_spec if self.cluster_spec.num_gpus > 1 else None,
            distribution_strategy=distribution_strategy,
            distributed_memory=distributed_memory,
            communication_analysis=communication_analysis,
            scaling_analysis=scaling_analysis,
            training_estimate=training_estimate
        )

    def _run_bottleneck_analysis(self, model_info: Dict,
                                distribution_strategy: Optional[DistributionStrategy] = None,
                                **kwargs) -> AnalysisResult:
        """Run bottleneck-focused analysis with multi-GPU considerations."""
        # First run progressive analysis as baseline
        logger.info("Running baseline progressive analysis")
        progressive_result = self._run_progressive_analysis(model_info, distribution_strategy, **kwargs)

        # Enhanced bottleneck detection
        logger.info("Stage 4/4: ... Enhanced Bottleneck Detection")
        bottlenecks = self._detect_bottlenecks(progressive_result)

        # Generate recommendations (pass complete result for better context)
        recommendations = self._generate_recommendations(bottlenecks, progressive_result)

        return AnalysisResult(
            analysis_type=AnalysisType.BOTTLENECK,
            analysis_stage=AnalysisStage.ENHANCED,
            model_info=progressive_result.model_info,
            memory_breakdown=progressive_result.memory_breakdown,
            access_patterns=progressive_result.access_patterns,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            bandwidth_metrics=progressive_result.bandwidth_metrics,
            cluster_spec=progressive_result.cluster_spec,
            distribution_strategy=progressive_result.distribution_strategy,
            distributed_memory=progressive_result.distributed_memory,
            communication_analysis=progressive_result.communication_analysis,
            scaling_analysis=progressive_result.scaling_analysis,
            training_estimate=progressive_result.training_estimate
        )

    def _calculate_bandwidth_metrics(self, workload_model, access_patterns) -> Dict:
        """Calculate comprehensive bandwidth metrics."""
        gpu_spec = self.gpu_spec

        metrics = {
            "theoretical_bandwidth_gb_s": gpu_spec.memory_bandwidth_gb_s,
            "effective_bandwidth_gb_s": gpu_spec.effective_bandwidth_gb_s,
            "memory_capacity_gb": gpu_spec.memory_capacity_gb
        }

        if access_patterns:
            metrics.update({
                "achieved_bandwidth_gb_s": (gpu_spec.effective_bandwidth_gb_s *
                                          access_patterns.bandwidth_utilization),
                "bandwidth_utilization": access_patterns.bandwidth_utilization,
                "sequential_ratio": access_patterns.sequential_ratio
            })

        # Add cluster-level metrics if multi-GPU
        if self.cluster_spec.num_gpus > 1:
            metrics.update({
                "cluster_total_bandwidth_gb_s": gpu_spec.memory_bandwidth_gb_s * self.cluster_spec.num_gpus,
                "interconnect_bandwidth_gb_s": self.cluster_spec.get_effective_bandwidth(),
                "cluster_memory_capacity_gb": self.cluster_spec.get_total_memory_capacity()
            })

        return metrics

    def _detect_bottlenecks(self, result: AnalysisResult) -> List[Dict]:
        """Detect memory and performance bottlenecks including multi-GPU considerations."""
        bottlenecks = []

        # Use distributed memory if available, otherwise single-GPU memory
        if result.distributed_memory:
            memory = result.distributed_memory["memory_per_gpu"]
            gpu_memory_gb = self.gpu_spec.memory_capacity_gb
            is_multi_gpu = True
        else:
            memory = result.memory_breakdown
            gpu_memory_gb = self.gpu_spec.memory_capacity_gb
            is_multi_gpu = False

        model_info = result.model_info

        # Memory capacity bottleneck (per GPU)
        memory_gb = memory.total_memory_gb if hasattr(memory, 'total_memory_gb') else memory["memory_per_gpu"].total_memory_gb

        if memory_gb > gpu_memory_gb * 0.9:
            severity = "critical" if memory_gb > gpu_memory_gb else "high"
            bottleneck_desc = f"Memory per GPU ({memory_gb:.1f} GB) exceeds 90% of capacity ({gpu_memory_gb} GB)"
            if is_multi_gpu:
                bottleneck_desc += f" in {self.cluster_spec.num_gpus}x GPU cluster"

            bottlenecks.append({
                "type": "memory_capacity",
                "severity": severity,
                "component": "total_memory",
                "description": bottleneck_desc,
                "impact": "Training may fail or require different parallelization strategy",
                "memory_gb": memory_gb,
                "gpu_capacity_gb": gpu_memory_gb,
                "is_multi_gpu": is_multi_gpu
            })

        # Multi-GPU specific bottlenecks
        if is_multi_gpu and result.communication_analysis:
            comm_overhead = result.communication_analysis.get("communication_overhead_percent", 0)

            if comm_overhead > 30:
                bottlenecks.append({
                    "type": "communication_overhead",
                    "severity": "high",
                    "component": "interconnect",
                    "description": f"High communication overhead ({comm_overhead:.1f}%) reduces scaling efficiency",
                    "impact": "Consider different parallelization strategy or faster interconnect",
                    "communication_overhead_percent": comm_overhead,
                    "interconnect": self.cluster_spec.interconnect
                })

            # Scaling efficiency bottleneck
            if result.scaling_analysis:
                efficiency = result.scaling_analysis.get("efficiency_percent", 100)
                if efficiency < 60:
                    bottlenecks.append({
                        "type": "scaling_efficiency",
                        "severity": "medium",
                        "component": "parallelization",
                        "description": f"Poor scaling efficiency ({efficiency:.1f}%) with {self.cluster_spec.num_gpus} GPUs",
                        "impact": "Multi-GPU setup may not be cost-effective",
                        "efficiency_percent": efficiency,
                        "num_gpus": self.cluster_spec.num_gpus
                    })

        # Add single-GPU bottlenecks (adapted for multi-GPU context)
        single_gpu_bottlenecks = self._detect_single_gpu_bottlenecks(result, memory, model_info)
        bottlenecks.extend(single_gpu_bottlenecks)

        return bottlenecks

    def _detect_single_gpu_bottlenecks(self, result: AnalysisResult, memory, model_info: Dict) -> List[Dict]:
        """Detect traditional single-GPU bottlenecks (adapted for multi-GPU context)."""
        bottlenecks = []

        # Extract memory values (handle both MemoryBreakdown and dict formats)
        if hasattr(memory, 'activations_gb'):
            activations_gb = memory.activations_gb
            model_params_gb = memory.model_parameters_gb
            optimizer_gb = memory.optimizer_states_gb
            total_memory_gb = memory.total_memory_gb
        else:
            memory_obj = memory["memory_per_gpu"]
            activations_gb = memory_obj.activations_gb
            model_params_gb = memory_obj.model_parameters_gb
            optimizer_gb = memory_obj.optimizer_states_gb
            total_memory_gb = memory_obj.total_memory_gb

        # Check for attention memory explosion (CRITICAL)
        if ("transformer" in model_info.get("model_type", "").lower() and
            model_info.get("sequence_length", 0) > 1024 and
            activations_gb > 100):  # Very high activation memory suggests attention issue

            sequence_length = model_info.get("sequence_length", 0)
            # Estimate what portion is likely attention memory
            # For long sequences, most activation memory is typically attention
            estimated_attention_gb = activations_gb * 0.8  # Conservative estimate

            severity = "critical" if activations_gb > 500 else "high"

            bottlenecks.append({
                "type": "attention_quadratic_memory",
                "severity": severity,
                "component": "attention_scores",
                "description": f"Massive activation memory ({activations_gb:.0f} GB) indicates quadratic attention scaling with {sequence_length} sequence length",
                "impact": "Standard attention is impractical for long sequences - Flash Attention required immediately",
                "memory_gb": activations_gb,
                "sequence_length": sequence_length,
                "attention_memory_gb": estimated_attention_gb,
                "scaling_type": "quadratic"
            })

        # Enhanced sequence length bottleneck (for transformers)
        if ("transformer" in model_info.get("model_type", "").lower() and
            model_info.get("sequence_length", 0) > 4096):

            sequence_length = model_info.get("sequence_length", 0)
            severity = "critical" if sequence_length > 16384 else "high"

            bottlenecks.append({
                "type": "sequence_length_quadratic",
                "severity": severity,
                "component": "attention",
                "description": f"Very long sequence ({sequence_length}) causes quadratic memory scaling in attention",
                "impact": "Consider Flash Attention, sequence reduction, or sparse attention patterns",
                "sequence_length": sequence_length,
                "memory_scaling": "quadratic"
            })

        # Traditional activation memory bottleneck (but lower priority if attention detected)
        activation_threshold = 2.0 if any(b.get("type") == "attention_quadratic_memory" for b in bottlenecks) else 1.5
        if activations_gb > model_params_gb * activation_threshold:
            bottlenecks.append({
                "type": "activation_memory",
                "severity": "high",
                "component": "activations",
                "description": f"Activation memory ({activations_gb:.1f} GB) is "
                            f"{activations_gb/model_params_gb:.1f}x larger than model parameters",
                "impact": "High memory usage, good candidate for activation checkpointing",
                "memory_gb": activations_gb,
                "ratio_to_params": activations_gb / model_params_gb
            })

        # Optimizer memory bottleneck
        if optimizer_gb > model_params_gb * 2.5:
            bottlenecks.append({
                "type": "optimizer_memory",
                "severity": "medium",
                "component": "optimizer_states",
                "description": f"Optimizer states ({optimizer_gb:.1f} GB) "
                            f"use {optimizer_gb/model_params_gb:.1f}x model parameter memory",
                "impact": "Consider memory-efficient optimizers like Adafactor or 8-bit Adam",
                "memory_gb": optimizer_gb,
                "ratio_to_params": optimizer_gb / model_params_gb
            })

        # Memory capacity bottleneck (catch-all for when total memory exceeds GPU capacity)
        gpu_capacity_gb = self.gpu_spec.memory_capacity_gb
        if total_memory_gb > gpu_capacity_gb * 0.9:
            severity = "critical" if total_memory_gb > gpu_capacity_gb else "high"
            overflow_gb = total_memory_gb - gpu_capacity_gb * 0.9

            bottlenecks.append({
                "type": "memory_capacity",
                "severity": severity,
                "component": "total_memory",
                "description": f"Total memory ({total_memory_gb:.1f} GB) exceeds 90% of GPU capacity ({gpu_capacity_gb} GB)",
                "impact": "Training will fail - requires optimization or multi-GPU setup",
                "memory_gb": total_memory_gb,
                "gpu_capacity_gb": gpu_capacity_gb,
                "overflow_gb": overflow_gb
            })

        # Batch size efficiency bottleneck (when batch size is very small)
        batch_size = model_info.get("batch_size", 1)
        if batch_size < 4 and total_memory_gb < gpu_capacity_gb * 0.5:
            bottlenecks.append({
                "type": "batch_size_efficiency",
                "severity": "low",
                "component": "batch_configuration",
                "description": f"Very small batch size ({batch_size}) with abundant GPU memory",
                "impact": "Consider increasing batch size for better GPU utilization",
                "batch_size": batch_size,
                "memory_utilization": total_memory_gb / gpu_capacity_gb
            })

        # Model architecture efficiency check
        if model_params_gb / total_memory_gb < 0.15:  # Model params < 15% of total memory
            bottlenecks.append({
                "type": "memory_efficiency",
                "severity": "medium",
                "component": "memory_allocation",
                "description": f"Model parameters ({model_params_gb:.1f} GB) are only {model_params_gb/total_memory_gb:.1%} of total memory",
                "impact": "High overhead suggests opportunity for optimization",
                "memory_efficiency": model_params_gb / total_memory_gb,
                "overhead_gb": total_memory_gb - model_params_gb
            })

        # Framework overhead bottleneck (when framework overhead is excessive)
        framework_overhead_gb = getattr(memory, 'framework_overhead_gb', 0) + getattr(memory, 'fragmentation_gb', 0)
        if framework_overhead_gb > model_params_gb * 0.3:  # Framework overhead > 30% of model size
            bottlenecks.append({
                "type": "framework_overhead",
                "severity": "low",
                "component": "framework",
                "description": f"Framework overhead ({framework_overhead_gb:.1f} GB) is high relative to model size",
                "impact": "Consider framework optimizations or memory management tuning",
                "overhead_gb": framework_overhead_gb,
                "overhead_ratio": framework_overhead_gb / model_params_gb
            })

        return bottlenecks

    def _generate_recommendations(self, bottlenecks: List[Dict],
                                result: AnalysisResult) -> List[Dict]:
        """Generate optimization recommendations using the sophisticated recommendations engine."""

        # Use the advanced recommendations engine
        from training_recommendations import get_optimization_recommendations

        # Prepare model info for the recommendations engine
        model_info = result.model_info.copy()
        model_info["is_multi_gpu"] = result.cluster_spec is not None
        if result.cluster_spec:
            model_info["num_gpus"] = result.cluster_spec.num_gpus
            model_info["gpu_type"] = result.cluster_spec.gpu_type

        # Use distributed memory if available, otherwise single GPU memory
        memory_usage = result.distributed_memory if result.distributed_memory else result.memory_breakdown

        # Call the sophisticated recommendations engine
        recommendations = get_optimization_recommendations(
            bottlenecks=bottlenecks,
            model_info=model_info,
            memory_usage=memory_usage,
            framework=self.framework
        )

        return recommendations

    def analyze_distributed_strategies(self, model_type: str, model_size_billions: float,
                                     batch_size: int, sequence_length: int = 2048) -> Dict:
        """Analyze multiple distribution strategies and recommend the best one."""
        if not self.distributed_analyzer:
            return {"error": "Multi-GPU cluster not configured"}

        # Create base memory breakdown with proper effective batch size
        effective_batch_size = max(1, batch_size / self.cluster_spec.num_gpus)  # Assume data parallel for base calculation
        workload_model = create_workload_model(
            model_type, model_size_billions, batch_size, sequence_length,
            self.gpu_type, effective_batch_size=effective_batch_size
        )
        base_memory = workload_model.calculate_memory_breakdown()

        # Get strategy recommendations
        recommendations = self.distributed_analyzer.recommend_optimal_strategy(
            base_memory, batch_size
        )

        return recommendations

    def get_max_batch_size(self, model_type: str, model_size_billions: float,
                          sequence_length: int = 2048, hidden_dim: Optional[int] = None,
                          memory_efficiency: float = 0.9,
                          distribution_strategy: Optional[DistributionStrategy] = None) -> int:
        """Estimate maximum batch size considering single or multi-GPU setup."""
        if distribution_strategy and self.distributed_analyzer:
            # Multi-GPU max batch size calculation
            # This is more complex as it depends on the parallelization strategy
            # For now, return a simple estimate
            single_gpu_max = estimate_max_batch_size(
                model_type, model_size_billions, sequence_length,
                self.gpu_type, memory_efficiency
            )

            if distribution_strategy.primary_strategy.value == "data_parallel":
                return single_gpu_max * distribution_strategy.data_parallel
            else:
                # For tensor/pipeline parallel, batch size doesn't scale linearly
                return single_gpu_max
        else:
            # Single GPU max batch size
            return estimate_max_batch_size(
                model_type, model_size_billions, sequence_length,
                self.gpu_type, memory_efficiency
            )

    def compare_configurations(self, model_type: str, model_size_billions: float,
                             batch_sizes: List[int] = None,
                             sequence_lengths: List[int] = None,
                             strategies: List[DistributionStrategy] = None) -> Dict:
        """Compare memory usage across different configurations including multi-GPU strategies."""
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64]
        if sequence_lengths is None:
            sequence_lengths = [512, 1024, 2048, 4096]

        results = {
            "batch_size_comparison": [],
            "sequence_length_comparison": [],
            "strategy_comparison": []
        }

        # Compare batch sizes
        for batch_size in batch_sizes:
            result = self.analyze_model(
                model_type, model_size_billions, batch_size, sequence_lengths[0]
            )

            memory_gb = (result.distributed_memory["memory_per_gpu"].total_memory_gb
                        if result.distributed_memory
                        else result.memory_breakdown.total_memory_gb)

            results["batch_size_comparison"].append({
                "batch_size": batch_size,
                "total_memory_gb": memory_gb,
                "is_multi_gpu": result.cluster_spec is not None
            })

        # Compare sequence lengths
        for seq_len in sequence_lengths:
            result = self.analyze_model(
                model_type, model_size_billions, batch_sizes[2], seq_len
            )

            memory_gb = (result.distributed_memory["memory_per_gpu"].total_memory_gb
                        if result.distributed_memory
                        else result.memory_breakdown.total_memory_gb)

            results["sequence_length_comparison"].append({
                "sequence_length": seq_len,
                "total_memory_gb": memory_gb,
                "is_multi_gpu": result.cluster_spec is not None
            })

        # Compare distribution strategies (if multi-GPU)
        if strategies and self.distributed_analyzer:
            for strategy in strategies:
                result = self.analyze_model(
                    model_type, model_size_billions, batch_sizes[2], sequence_lengths[1],
                    distribution_strategy=strategy
                )

                if result.distributed_memory:
                    results["strategy_comparison"].append({
                        "strategy": strategy.to_dict(),
                        "memory_per_gpu_gb": result.distributed_memory["memory_per_gpu"].total_memory_gb,
                        "total_cluster_memory_gb": result.distributed_memory["total_cluster_memory"],
                        "communication_overhead_percent": result.communication_analysis.get("communication_overhead_percent", 0),
                        "scaling_efficiency_percent": result.scaling_analysis.get("efficiency_percent", 0)
                    })

        return results