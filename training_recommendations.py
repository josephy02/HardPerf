import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Optimization targets for recommendations
class OptimizationTarget:
    """Optimization target constants."""
    MEMORY = "memory"
    BANDWIDTH = "bandwidth"
    CACHE = "cache"
    THROUGHPUT = "throughput"

@dataclass
class Recommendation:
    """Single optimization recommendation."""
    title: str
    description: str
    techniques: List[str]
    priority: str  # critical, high, medium, low
    difficulty: str  # low, medium, high
    estimated_savings: str
    target: str  # memory, bandwidth, cache, throughput
    applies_to: List[str] = None  # Model types this applies to

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "techniques": self.techniques,
            "priority": self.priority,
            "difficulty": self.difficulty,
            "estimated_savings": self.estimated_savings,
            "target": self.target,
            "applies_to": self.applies_to or []
        }

class RecommendationEngine:
    """Generates optimization recommendations focused on fixed infrastructure constraints."""

    def __init__(self, framework: str = "pytorch", fixed_infrastructure: bool = True):
        """Initialize recommendation engine."""
        self.framework = framework
        self.fixed_infrastructure = fixed_infrastructure
        self._init_recommendation_templates()

    def _init_recommendation_templates(self):
        """Initialize recommendation templates focused on software optimizations."""
        self.memory_recommendations = {
            "memory_capacity": Recommendation(
                title="Reduce Memory Usage (Required)",
                description="Model memory requirements exceed available GPU capacity",
                techniques=[
                    "Enable gradient checkpointing (60-80% activation memory reduction)",
                    "Use mixed precision (FP16/BF16) training (50% total memory reduction)",
                    "Reduce batch size and use gradient accumulation",
                    "Enable Flash Attention for transformers (90%+ attention memory reduction)"
                ],
                priority="critical",
                difficulty="low",
                estimated_savings="60-80% memory reduction with gradient checkpointing",
                target=OptimizationTarget.MEMORY
            ),

            "activation_memory": Recommendation(
                title="Optimize Activation Memory Usage",
                description="Activation tensors consume excessive memory during forward pass",
                techniques=[
                    "Enable gradient checkpointing for transformer blocks",
                    "Use selective checkpointing for memory-intensive layers only",
                    "Apply activation compression techniques if available",
                    "Consider reducing sequence length if model quality permits"
                ],
                priority="high",
                difficulty="low",
                estimated_savings="50-80% activation memory reduction",
                target=OptimizationTarget.MEMORY,
                applies_to=["transformer", "vision_transformer"]
            ),

            "attention_quadratic_memory": Recommendation(
                title="CRITICAL: Enable Flash Attention for Long Sequences",
                description="Standard attention requires quadratic memory with sequence length, causing massive memory usage",
                techniques=[
                    "Enable Flash Attention to change O(S²) to O(S) memory scaling",
                    "Consider sparse attention patterns as alternative",
                    "Reduce sequence length if Flash Attention unavailable"
                ],
                priority="critical",
                difficulty="low",
                estimated_savings="90-95% attention memory reduction (changes O(S²) to O(S))",
                target=OptimizationTarget.MEMORY,
                applies_to=["transformer", "vision_transformer"]
            ),

            "sequence_length_quadratic": Recommendation(
                title="Optimize for Quadratic Attention Scaling",
                description="Long sequences cause quadratic memory scaling in standard attention",
                techniques=[
                    "Enable Flash Attention to change O(S²) to O(S) scaling",
                    "Use sparse attention patterns for very long sequences",
                    "Consider sliding window attention with local context",
                    "Reduce sequence length to 1024 or 512 if Flash Attention unavailable"
                ],
                priority="critical",
                difficulty="medium",
                estimated_savings="Massive reduction: from TB to GB scale for long sequences",
                target=OptimizationTarget.MEMORY,
                applies_to=["transformer", "vision_transformer"]
            ),

            "optimizer_memory": Recommendation(
                title="Use Memory-Efficient Optimizers",
                description="Optimizer states (momentum, variance) consume excessive memory",
                techniques=[
                    "Switch to 8-bit Adam optimizer (saves 75% optimizer memory)",
                    "Use Adafactor for large embedding layers",
                    "Consider Lion optimizer for reduced memory requirements",
                    "Enable optimizer state CPU offloading if supported"
                ],
                priority="medium",
                difficulty="low",
                estimated_savings="50-75% optimizer memory reduction",
                target=OptimizationTarget.MEMORY
            ),

            "sequence_length": Recommendation(
                title="Optimize Long Sequence Processing",
                description="Long sequences cause quadratic memory scaling in attention",
                techniques=[
                    "Enable Flash Attention for memory-efficient attention computation",
                    "Use sparse attention patterns (sliding window, local attention)",
                    "Consider sequence length reduction if task permits",
                    "Apply gradient checkpointing specifically for attention layers"
                ],
                priority="high",
                difficulty="medium",
                estimated_savings="60-90% attention memory reduction",
                target=OptimizationTarget.MEMORY,
                applies_to=["transformer", "vision_transformer"]
            )
        }

        self.multi_gpu_recommendations = {
            "communication_overhead": Recommendation(
                title="Reduce Multi-GPU Communication Overhead",
                description="High communication costs reduce distributed training efficiency",
                techniques=[
                    "Use gradient compression to reduce communication volume",
                    "Enable mixed precision (FP16) for gradient communication",
                    "Optimize batch size per GPU to reduce communication frequency",
                    "Consider switching parallelization strategy if overhead is extreme"
                ],
                priority="high",
                difficulty="medium",
                estimated_savings="20-50% reduction in communication overhead",
                target=OptimizationTarget.THROUGHPUT
            ),

            "scaling_efficiency": Recommendation(
                title="Improve Multi-GPU Scaling Efficiency",
                description="Poor scaling efficiency with current GPU configuration",
                techniques=[
                    "Increase batch size per GPU to improve compute-to-communication ratio",
                    "Enable gradient accumulation to maintain effective batch size",
                    "Optimize data loading pipeline to prevent GPU starvation",
                    "Consider different parallelization strategy for better efficiency"
                ],
                priority="medium",
                difficulty="medium",
                estimated_savings="10-30% improvement in GPU utilization",
                target=OptimizationTarget.THROUGHPUT
            )
        }

        self.framework_recommendations = {
            "pytorch": [
                Recommendation(
                    title="Enable PyTorch Memory Optimizations",
                    description="Use PyTorch-specific optimizations for better memory efficiency",
                    techniques=[
                        "Enable torch.compile for automatic optimization (PyTorch 2.0+)",
                        "Use torch.utils.checkpoint for gradient checkpointing",
                        "Enable mixed precision with torch.cuda.amp.autocast",
                        "Use Flash Attention from torch.nn.functional.scaled_dot_product_attention"
                    ],
                    priority="medium",
                    difficulty="low",
                    estimated_savings="10-30% performance improvement",
                    target=OptimizationTarget.THROUGHPUT
                )
            ],

            "tensorflow": [
                Recommendation(
                    title="Enable TensorFlow Memory Optimizations",
                    description="Use TensorFlow-specific optimizations for better efficiency",
                    techniques=[
                        "Enable XLA compilation with tf.function(jit_compile=True)",
                        "Use mixed precision with tf.keras.mixed_precision",
                        "Apply gradient checkpointing with tf.recompute_grad",
                        "Enable memory growth with tf.config.experimental.set_memory_growth"
                    ],
                    priority="medium",
                    difficulty="low",
                    estimated_savings="15-35% performance improvement",
                    target=OptimizationTarget.THROUGHPUT
                )
            ]
        }

    def _get_total_memory(self, memory_usage) -> float:
        """Extract total memory from memory usage object."""
        if hasattr(memory_usage, 'total_memory_gb'):
            return memory_usage.total_memory_gb
        elif isinstance(memory_usage, dict) and 'memory_per_gpu' in memory_usage:
            return memory_usage['memory_per_gpu'].total_memory_gb
        return 0.0


    def _generate_multi_gpu_recommendation(self, overflow_gb: float,
                                        model_info: Dict, total_memory_gb: float) -> Dict:
        """Generate multi-GPU strategy recommendation based on memory overflow."""
        model_size_gb = model_info.get("model_size_billions", 1.0)
        gpu_type = model_info.get("gpu_type", "NVIDIA H100")

        # Get actual GPU capacity for the specific GPU type
        from gpu_specs import get_gpu_memory_capacity
        gpu_capacity_total = get_gpu_memory_capacity(gpu_type)
        gpu_capacity_usable = gpu_capacity_total * 0.9  # 90% usable capacity
        target_memory_per_gpu = gpu_capacity_usable * 0.8  # Target 80% utilization

        # Data parallel: each GPU needs full model + activations/batch split
        model_memory = model_size_gb * 4  # params + optimizer + gradients
        activation_memory = total_memory_gb - model_memory

        # For data parallel: model memory stays same, activations split
        data_parallel_gpus = max(2, int(activation_memory / (target_memory_per_gpu - model_memory)) + 1)

        # For tensor parallel: everything scales down
        tensor_parallel_gpus = max(2, int(total_memory_gb / target_memory_per_gpu) + 1)

        # Choose the better strategy based on model size and GPU requirements
        if data_parallel_gpus <= 8 and model_size_gb <= 15:
            strategy = "data_parallel"
            gpus_needed = data_parallel_gpus
            efficiency = "85-95%"
            memory_per_gpu = model_memory + activation_memory / gpus_needed
        else:
            strategy = "tensor_parallel"
            gpus_needed = tensor_parallel_gpus
            efficiency = "60-80%"
            memory_per_gpu = total_memory_gb / gpus_needed * 1.15  # 15% communication overhead

        # Ensure we don't recommend impossible configurations
        max_iterations = 0
        while memory_per_gpu > gpu_capacity_usable and gpus_needed < 32 and max_iterations < 10:
            gpus_needed += 1
            if strategy == "data_parallel":
                memory_per_gpu = model_memory + activation_memory / gpus_needed
            else:
                memory_per_gpu = total_memory_gb / gpus_needed * 1.15
            max_iterations += 1

        # Update strategy for large clusters
        if gpus_needed > 8:
            if gpus_needed <= 16:
                strategy = "tensor_parallel"
                efficiency = "60-80%"
            else:
                strategy = "hybrid (tensor + pipeline parallel)"
                efficiency = "40-60%"

        # Create GPU-specific messaging
        gpu_short_name = gpu_type.replace("NVIDIA ", "")

        return {
            "title": f"Use Multi-GPU Training ({gpus_needed}x {gpu_short_name})",
            "description": f"Single {gpu_short_name} cannot fit {total_memory_gb:.0f} GB model - distribute across multiple GPUs",
            "techniques": [
                f"Use {strategy.replace('_', ' ')} strategy with {gpus_needed} {gpu_short_name} GPUs",
                f"Expected memory per GPU: {memory_per_gpu:.1f} GB",
                f"Training efficiency: {efficiency}"
            ],
            "priority": "high",
            "difficulty": "medium",
            "estimated_savings": f"Reduce from {total_memory_gb:.0f} GB to {memory_per_gpu:.1f} GB per GPU",
            "target": "memory"
        }


    def _generate_combined_savings_estimate(self, bottlenecks: List[Dict],
                                        total_memory_gb: float, model_info: Dict) -> Dict:
        """Generate estimate of combined optimization savings."""

        # Get GPU-specific information
        gpu_type = model_info.get("gpu_type", "NVIDIA H100")
        model_size_gb = model_info.get("model_size_billions", 1.0)

        # Get actual GPU capacity for the specific GPU type
        from gpu_specs import get_gpu_memory_capacity
        gpu_capacity_total = get_gpu_memory_capacity(gpu_type)
        gpu_capacity_usable = gpu_capacity_total * 0.9  # 90% usable capacity
        target_memory = gpu_capacity_usable * 0.8  # Target 80% utilization

        # Calculate potential savings from each optimization
        flash_attention_savings = 0
        gradient_checkpoint_savings = 0

        for bottleneck in bottlenecks:
            bottleneck_type = bottleneck.get("type", "")

            if "attention" in bottleneck_type:
                # Flash Attention can reduce attention memory by 90%
                attention_memory = bottleneck.get("attention_memory_gb", 0)
                flash_attention_savings = max(flash_attention_savings, attention_memory * 0.9)
            elif "activation" in bottleneck_type:
                # Gradient checkpointing reduces activation memory by 60%
                activation_memory = bottleneck.get("memory_gb", 0)
                gradient_checkpoint_savings = max(gradient_checkpoint_savings, activation_memory * 0.6)

        # Apply optimizations sequentially
        # First apply Flash Attention (biggest impact)
        memory_after_flash = total_memory_gb - flash_attention_savings

        # Then apply gradient checkpointing to remaining activations
        # Note: gradient checkpointing affects the remaining activation memory, not the original
        remaining_activation_memory = memory_after_flash * 0.8  # Assume 80% is still activations
        actual_gradient_savings = min(gradient_checkpoint_savings, remaining_activation_memory * 0.6)

        optimized_memory = memory_after_flash - actual_gradient_savings
        # Ensure minimum reasonable memory (model + optimizer + gradients + minimal activations)
        min_memory = model_size_gb * 4 + 5  # params + optimizer + gradients + 5GB minimal activations
        optimized_memory = max(optimized_memory, min_memory)

        # Multi-GPU calculation using actual GPU capacity
        if optimized_memory > target_memory:
            gpus_needed = max(1, int(optimized_memory / target_memory) + 1)
            final_memory_per_gpu = optimized_memory / gpus_needed * 1.1  # 10% overhead for multi-GPU

            # Ensure we don't exceed GPU capacity even with overhead
            max_iterations = 0
            while final_memory_per_gpu > gpu_capacity_usable and gpus_needed < 16 and max_iterations < 10:
                gpus_needed += 1
                final_memory_per_gpu = optimized_memory / gpus_needed * 1.1
                max_iterations += 1

            # Determine strategy based on GPU count
            if gpus_needed <= 4:
                strategy = "data_parallel"
            elif gpus_needed <= 8:
                strategy = "tensor_parallel"
            else:
                strategy = "hybrid"
        else:
            gpus_needed = 1
            final_memory_per_gpu = optimized_memory
            strategy = "single_gpu"

        # Build techniques list with GPU-specific messaging
        techniques = []
        if flash_attention_savings > 0:
            techniques.append(f"Flash Attention: -{flash_attention_savings:.0f} GB")
        if actual_gradient_savings > 0:
            techniques.append(f"Gradient Checkpointing: -{actual_gradient_savings:.0f} GB")
        if gpus_needed > 1:
            gpu_short_name = gpu_type.replace("NVIDIA ", "")
            techniques.append(f"Multi-GPU ({gpus_needed}x {gpu_short_name}): {strategy.replace('_', ' ').title()}")

        if not techniques:
            techniques = ["Apply recommended memory optimizations"]

        # Create GPU-specific result message
        if gpus_needed == 1:
            gpu_short_name = gpu_type.replace("NVIDIA ", "")
            gpu_message = f"single {gpu_short_name}"
        else:
            gpu_short_name = gpu_type.replace("NVIDIA ", "")
            gpu_message = f"{gpus_needed}x {gpu_short_name} GPUs needed"

        result = {
            "title": "Combined Optimization Impact",
            "description": f"Expected results after applying all recommended optimizations on {gpu_type}",
            "techniques": techniques,
            "priority": "info",
            "difficulty": "info",
            "estimated_savings": f"Final memory requirement: {final_memory_per_gpu:.1f} GB per GPU ({gpu_message})",
            "target": "summary"
        }

        return result



    def generate_recommendations(self, bottlenecks: List[Dict],
                            model_info: Dict, memory_usage: Dict) -> List[Dict]:
        """Generate recommendations focused on fixed infrastructure constraints."""
        recommendations = []
        model_type = model_info.get("model_type", "").lower()
        is_multi_gpu = model_info.get("is_multi_gpu", False)
        gpu_type = model_info.get("gpu_type", "Unknown")
        num_gpus = model_info.get("num_gpus", 1)
        sequence_length = model_info.get("sequence_length", 0)

        # Get total memory
        total_memory = self._get_total_memory(memory_usage)

        # Track what has been added to avoid duplicates
        flash_attention_added = False
        memory_capacity_added = False

        # Check for attention memory crisis (single critical Flash Attention recommendation)
        if (total_memory > 500 and  # >500GB suggests attention memory explosion
            "transformer" in model_type and
            sequence_length > 1024):

            flash_attention_critical = {
                "title": "CRITICAL: Enable Flash Attention for Long Sequences",
                "description": f"Standard attention with {sequence_length} sequence length requires {total_memory:.0f}GB memory. This is impractical without Flash Attention.",
                "techniques": [
                    "Enable Flash Attention to change O(S²) to O(S) memory scaling",
                    "Alternative: Reduce sequence length to 512-1024 tokens"
                ],
                "priority": "critical",
                "difficulty": "low",
                "estimated_savings": f"Reduce from {total_memory:.0f}GB to ~{total_memory * 0.1:.0f}GB (90%+ reduction)",
                "target": OptimizationTarget.MEMORY
            }
            recommendations.append(flash_attention_critical)
            flash_attention_added = True

        # Infrastructure constraint context
        if self.fixed_infrastructure:
            infrastructure_note = {
                "title": "Infrastructure Optimization Context",
                "description": f"Optimizing for your existing {num_gpus}x {gpu_type} setup",
                "techniques": [
                    "Focus: Software and algorithm optimizations within hardware constraints",
                    "Goal: Make training feasible on current infrastructure",
                    "Approach: Memory reduction and efficiency improvements"
                ],
                "priority": "info",
                "difficulty": "info",
                "estimated_savings": "Recommendations tailored to work within your hardware limits",
                "target": "infrastructure_context"
            }
            recommendations.append(infrastructure_note)

        # Generate recommendations for each bottleneck (with deduplication)
        for i, bottleneck in enumerate(bottlenecks):
            bottleneck_type = bottleneck.get("type", "")

            # Skip attention-related bottlenecks if Flash Attention already added
            if flash_attention_added and ("attention" in bottleneck_type or "sequence_length" in bottleneck_type):
                continue

            # Skip memory capacity if we'll add multi-GPU recommendation
            if bottleneck_type == "memory_capacity" and total_memory > 200:  # Will get multi-GPU rec
                memory_capacity_added = True
                continue

            # Get base recommendation for other bottleneck types
            rec = None
            if bottleneck_type in self.memory_recommendations:
                rec = self.memory_recommendations[bottleneck_type]
            elif bottleneck_type in self.multi_gpu_recommendations:
                rec = self.multi_gpu_recommendations[bottleneck_type]

            if rec:
                # Check if recommendation applies to this model type
                if rec.applies_to and not any(applies in model_type for applies in rec.applies_to):
                    continue

                # Customize recommendation based on specific bottleneck data
                customized_rec = self._customize_recommendation(rec, bottleneck, model_info)
                recommendations.append(customized_rec.to_dict())

        # Add multi-GPU recommendation for significant memory overflow
        gpu_capacity = 72  # H100 usable capacity
        memory_overflow_gb = total_memory - gpu_capacity

        if memory_overflow_gb > 20:  # Significant overflow
            multi_gpu_rec = self._generate_multi_gpu_recommendation(
                memory_overflow_gb, model_info, total_memory
            )
            if multi_gpu_rec:
                recommendations.append(multi_gpu_rec)

        # Framework-specific recommendations
        if self.framework in self.framework_recommendations:
            for rec in self.framework_recommendations[self.framework]:
                recommendations.append(rec.to_dict())

        # Add combined savings estimate at the end if multiple recommendations
        non_info_recs = [r for r in recommendations if r.get("priority") not in ["info"]]
        print(f"Debug: non_info_recs count = {len(non_info_recs)}")
        if len(non_info_recs) > 1:
            combined_rec = self._generate_combined_savings_estimate(
                bottlenecks, total_memory, model_info
            )
            recommendations.append(combined_rec)

        # General optimization recommendations if no specific bottlenecks
        if len(non_info_recs) == 0:
            general_recs = self._get_general_recommendations(model_info, memory_usage)
            recommendations.extend(general_recs)

        # Sort by priority and remove duplicates
        recommendations = self._prioritize_and_deduplicate(recommendations)

        return recommendations

    def _customize_recommendation(self, rec: Recommendation, bottleneck: Dict,
                                 model_info: Dict) -> Recommendation:
        """Customize recommendation based on specific bottleneck details."""
        # Create a copy to avoid modifying the template
        customized = Recommendation(
            title=rec.title,
            description=rec.description,
            techniques=rec.techniques.copy(),
            priority=rec.priority,
            difficulty=rec.difficulty,
            estimated_savings=rec.estimated_savings,
            target=rec.target,
            applies_to=rec.applies_to
        )

        # Customize based on bottleneck severity
        severity = bottleneck.get("severity", "medium")
        if severity == "critical":
            customized.priority = "critical"

        # Enhanced customization for attention memory bottlenecks
        if "attention" in bottleneck.get("type", ""):
            attention_memory = bottleneck.get("attention_memory_gb", 0)
            sequence_length = model_info.get("sequence_length", 0)

            if attention_memory > 100:  # Very high attention memory
                customized.priority = "critical"
                customized.estimated_savings = f"Reduce attention memory from {attention_memory:.0f}GB to ~{attention_memory * 0.1:.0f}GB"

                # Add specific Flash Attention guidance
                flash_guidance = f"For {sequence_length} tokens: Flash Attention is mandatory, not optional"
                if flash_guidance not in customized.techniques:
                    customized.techniques.insert(0, flash_guidance)

        # Specific savings estimates if available
        if "memory_gb" in bottleneck:
            memory_gb = bottleneck["memory_gb"]
            if rec.target == "memory":
                if "activation" in bottleneck.get("type", ""):
                    savings_range = f"{memory_gb * 0.6:.1f}-{memory_gb * 0.8:.1f} GB"
                    customized.estimated_savings = f"Save {savings_range} with gradient checkpointing"
                elif "capacity" in bottleneck.get("type", ""):
                    overflow = memory_gb - bottleneck.get("gpu_capacity_gb", memory_gb)
                    customized.estimated_savings = f"Reduce {overflow:.1f} GB overflow with optimizations"

        # Model-specific techniques
        model_type = model_info.get("model_type", "").lower()
        if "transformer" in model_type and rec.target == "memory":
            if not any("Flash Attention" in technique for technique in customized.techniques):
                customized.techniques.append("Enable Flash Attention for transformer models")

        # Multi-GPU specific context
        if model_info.get("is_multi_gpu", False):
            num_gpus = model_info.get("num_gpus", 1)
            if "memory" in rec.target and num_gpus > 1:
                customized.description += f" (per GPU in {num_gpus}x GPU setup)"

        return customized

    def _get_general_recommendations(self, model_info: Dict, memory_usage: Dict) -> List[Dict]:
        """Get general optimization recommendations when no specific bottlenecks found."""
        recommendations = []

        # Always recommend mixed precision for better efficiency
        recommendations.append({
            "title": "Enable Mixed Precision Training",
            "description": "Use FP16/BF16 for better memory efficiency and performance",
            "techniques": [
                "Enable automatic mixed precision in your training framework",
                "Use BF16 for better numerical stability with large models",
                "Monitor for gradient underflow and adjust loss scaling if needed"
            ],
            "priority": "medium",
            "difficulty": "low",
            "estimated_savings": "30-50% memory reduction, 20-40% speedup",
            "target": OptimizationTarget.MEMORY
        })

        # Recommend gradient checkpointing for larger models
        model_size = model_info.get("model_size_billions", 0)
        if model_size > 1.0:
            recommendations.append({
                "title": "Consider Gradient Checkpointing",
                "description": "Trade compute for memory by recomputing activations during backward pass",
                "techniques": [
                    "Enable gradient checkpointing for transformer blocks",
                    "Use selective checkpointing to balance memory vs compute trade-off",
                    "Monitor training speed impact (typically 20-30% slower)"
                ],
                "priority": "low",
                "difficulty": "low",
                "estimated_savings": f"Significant activation memory reduction for {model_size}B model",
                "target": OptimizationTarget.MEMORY
            })

        # Flash Attention recommendation for transformers with long sequences
        sequence_length = model_info.get("sequence_length", 0)
        if ("transformer" in model_info.get("model_type", "").lower() and
            sequence_length > 1024):
            recommendations.append({
                "title": "Enable Flash Attention for Long Sequences",
                "description": f"Sequence length {sequence_length} benefits significantly from Flash Attention",
                "techniques": [
                    "Enable Flash Attention to reduce quadratic attention memory scaling",
                    "Use built-in Flash Attention in PyTorch 2.0+ or Transformers library"
                ],
                "priority": "high",
                "difficulty": "low",
                "estimated_savings": "60-90% attention memory reduction",
                "target": OptimizationTarget.MEMORY
            })

        return recommendations

    def _prioritize_and_deduplicate(self, recommendations: List[Dict]) -> List[Dict]:
        """Sort recommendations by priority and remove duplicates."""
        # Remove duplicates based on title
        seen_titles = set()
        unique_recommendations = []

        for rec in recommendations:
            title = rec.get("title", "")
            if title not in seen_titles:
                unique_recommendations.append(rec)
                seen_titles.add(title)

        # Sort by priority (critical first, then info, then others)
        def priority_sort_key(rec):
            priority = rec.get("priority", "low")
            if priority == "critical":
                return -2  # Critical messages first
            elif priority == "info":
                return -1  # Info messages second
            priority_order = {"high": 1, "medium": 2, "low": 3}
            return priority_order.get(priority, 4)

        unique_recommendations.sort(key=priority_sort_key)

        return unique_recommendations

    def get_quick_wins(self, recommendations: List[Dict]) -> List[Dict]:
        """Get recommendations that are high impact and low difficulty."""
        quick_wins = []

        for rec in recommendations:
            priority = rec.get("priority", "low")
            difficulty = rec.get("difficulty", "high")

            # Critical/high priority + low difficulty = quick win
            if priority in ["critical", "high"] and difficulty == "low":
                quick_wins.append(rec)

        return quick_wins

    def generate_fixed_infrastructure_summary(self, model_info: Dict,
                                            memory_analysis: Dict) -> Dict[str, str]:
        """Generate summary focused on working within infrastructure constraints."""
        gpu_type = model_info.get("gpu_type", "Unknown")
        num_gpus = model_info.get("num_gpus", 1)
        model_size = model_info.get("model_size_billions", 0)

        # Get memory requirements
        if "distributed_memory" in memory_analysis and memory_analysis["distributed_memory"]:
            memory_per_gpu = memory_analysis["distributed_memory"]["memory_per_gpu"].total_memory_gb
        else:
            memory_per_gpu = memory_analysis.get("memory_breakdown", {}).get("total_memory_gb", 0)

        # Get GPU capacity
        from gpu_specs import get_gpu_memory_capacity
        gpu_capacity = get_gpu_memory_capacity(gpu_type)

        # Determine feasibility and required actions
        if memory_per_gpu <= gpu_capacity * 0.9:
            feasibility = "✓ Configuration will work with current hardware"
            action_needed = "No immediate action required"
        elif memory_per_gpu <= gpu_capacity * 1.2:
            feasibility = "⚠ Marginal fit - optimization recommended"
            action_needed = "Apply memory optimizations for reliable training"
        elif memory_per_gpu > gpu_capacity * 10:  # Extreme case (like our attention issue)
            feasibility = "🚨 CRITICAL: Requires immediate optimization"
            action_needed = "Enable Flash Attention and other critical optimizations immediately"
        else:
            feasibility = "✗ Requires optimization to fit in current hardware"
            action_needed = "Apply multiple memory optimizations (gradient checkpointing + mixed precision)"

        summary = {
            "infrastructure_context": f"Analysis for {num_gpus}x {gpu_type} setup",
            "feasibility_assessment": feasibility,
            "action_required": action_needed,
            "optimization_focus": "Software and algorithm optimizations within hardware constraints",
            "memory_status": f"{memory_per_gpu:.1f} GB required per GPU, {gpu_capacity} GB available"
        }

        return summary

# Utility functions for backward compatibility
def get_optimization_recommendations(bottlenecks: List[Dict], model_info: Dict,
                                   memory_usage: Dict, framework: str = "pytorch") -> List[Dict]:
    """Get optimization recommendations focused on fixed infrastructure (utility function)."""
    engine = RecommendationEngine(framework=framework, fixed_infrastructure=True)
    return engine.generate_recommendations(bottlenecks, model_info, memory_usage)

def get_quick_optimization_wins(recommendations: List[Dict]) -> List[Dict]:
    """Get quick optimization wins (utility function)."""
    engine = RecommendationEngine()
    return engine.get_quick_wins(recommendations)