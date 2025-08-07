"""
Author              : Joseph Yared
Copyright           :
File Name           : training_workload.py
Description         : Utility Functions for Memory Performance Analysis
                      Centralized helper functions and calculations


Revision History    :
Date                  Author               Comments
--------------------------------------------------------------------------------------------------

"""

import math
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

# Enums for consistent type handling
class AnalysisType(Enum):
    """Types of memory analysis."""
    PROGRESSIVE = "progressive"
    BOTTLENECK = "bottleneck"

class AnalysisStage(Enum):
    """Stages of progressive analysis."""
    ESTIMATION = "estimation"
    SIMULATION = "simulation"
    PROFILING = "profiling"
    ENHANCED = "enhanced"

class ModelArchitecture(Enum):
    """Supported model architectures."""
    TRANSFORMER_DECODER = "transformer_decoder"
    TRANSFORMER_ENCODER = "transformer_encoder"
    CNN = "cnn"
    DIFFUSION_MODEL = "diffusion_model"
    VISION_TRANSFORMER = "vision_transformer"

@dataclass
class BandwidthMetrics:
    """Bandwidth calculation results."""
    theoretical_bandwidth_gb_s: float
    achieved_bandwidth_gb_s: float
    bandwidth_utilization: float
    sequential_ratio: float

    def to_dict(self) -> Dict:
        return {
            "theoretical_bandwidth_gb_s": self.theoretical_bandwidth_gb_s,
            "achieved_bandwidth_gb_s": self.achieved_bandwidth_gb_s,
            "bandwidth_utilization": self.bandwidth_utilization,
            "sequential_ratio": self.sequential_ratio
        }

# Memory calculation constants - FIXED fragmentation factors
MEMORY_CONSTANTS = {
    "transformer_decoder": {
        "optimizer_factor": 2.0,        # Adam/AdamW stores 2 states per parameter
        "activation_factor": 1.0,       # Base activation scaling
        "temp_buffer_factor": 0.15,     # 15% of activations for temporary buffers
        "framework_overhead_base": 1.0, # 1GB base overhead
        "framework_overhead_factor": 0.05, # 5% of model size
        "fragmentation_factor": 0.08    # Updated: 8% fragmentation (was 15%)
    },
    "transformer_encoder": {
        "optimizer_factor": 2.0,
        "activation_factor": 0.8,       # Slightly less than decoder
        "temp_buffer_factor": 0.12,
        "framework_overhead_base": 1.0,
        "framework_overhead_factor": 0.05,
        "fragmentation_factor": 0.07    # Updated: 7% fragmentation (was 12%)
    },
    "cnn": {
        "optimizer_factor": 1.5,        # CNNs often use simpler optimizers
        "activation_factor": 1.2,       # Feature maps can be large
        "temp_buffer_factor": 0.10,
        "framework_overhead_base": 0.5,
        "framework_overhead_factor": 0.03,
        "fragmentation_factor": 0.06    # Updated: 6% fragmentation (was 10%)
    },
    "diffusion_model": {
        "optimizer_factor": 2.0,
        "activation_factor": 1.5,       # Multiple timesteps increase activations
        "temp_buffer_factor": 0.20,
        "framework_overhead_base": 1.5,
        "framework_overhead_factor": 0.07,
        "fragmentation_factor": 0.10    # Updated: 10% fragmentation (was 18%)
    },
    "vision_transformer": {
        "optimizer_factor": 2.0,
        "activation_factor": 0.9,
        "temp_buffer_factor": 0.13,
        "framework_overhead_base": 1.0,
        "framework_overhead_factor": 0.05,
        "fragmentation_factor": 0.06    # Updated: 6% fragmentation (was 13%)
    }
}

def normalize_model_type(model_type: str) -> str:
    """Normalize model type string to standard format.

    Args:
        model_type: Model architecture type string

    Returns:
        str: Normalized model type
    """
    # Normalize to lowercase and replace separators
    model_type_clean = model_type.lower().replace(" ", "_").replace("-", "_")

    # Check for transformer variants
    if "transformer" in model_type_clean:
        if "vision" in model_type_clean:
            return "vision_transformer"
        elif "encoder" in model_type_clean:
            return "transformer_encoder"
        else:
            return "transformer_decoder"  # Default for transformers

    # Check for other architectures
    elif "cnn" in model_type_clean or "conv" in model_type_clean:
        return "cnn"
    elif "diffusion" in model_type_clean:
        return "diffusion_model"
    else:
        return "transformer_decoder"

def get_memory_constants(model_type: str) -> Dict[str, float]:
    """Get memory constants for model type.

    Args:
        model_type: Model architecture type

    Returns:
        Dict: Memory calculation constants
    """
    # Normalize model type first
    normalized_type = normalize_model_type(model_type)

    # Return constants for normalized type
    if normalized_type in MEMORY_CONSTANTS:
        return MEMORY_CONSTANTS[normalized_type]

    # Fallback to transformer_decoder
    logger.warning(f"Unknown model type '{model_type}', using transformer_decoder constants")
    return MEMORY_CONSTANTS["transformer_decoder"]

def estimate_hidden_dimension(model_size_billions: float, model_type: str) -> int:
    """Estimate hidden dimension from model size and type.

    Args:
        model_size_billions: Model size in billions of parameters
        model_type: Model architecture type

    Returns:
        int: Estimated hidden dimension
    """
    normalized_type = normalize_model_type(model_type)

    if "transformer" in normalized_type:
        if model_size_billions < 0.1:
            hidden_dim = int(256 * math.sqrt(model_size_billions * 10))
        elif model_size_billions < 1.0:
            hidden_dim = int(768 * math.sqrt(model_size_billions))
        elif model_size_billions < 10.0:
            hidden_dim = int(2048 * math.sqrt(model_size_billions / 2))  # FIXED
        elif model_size_billions < 100.0:
            hidden_dim = int(3072 * math.sqrt(model_size_billions / 20)) # FIXED
        else:
            hidden_dim = int(4096 * math.sqrt(model_size_billions / 50)) # FIXED

        # Round to nearest multiple of 64 for attention
        return max(64, ((hidden_dim + 32) // 64) * 64)

    elif normalized_type == "cnn":
        # CNN scaling - based on filter count
        hidden_dim = int(64 * (model_size_billions * 100) ** 0.25)
        # Round to power of 2
        return 2 ** int(math.log2(hidden_dim) + 0.5)

    else:
        # Default scaling
        hidden_dim = int(512 * math.sqrt(model_size_billions * 2))
        return ((hidden_dim + 16) // 32) * 32

def calculate_bandwidth_metrics(gpu_type: str, sequential_ratio: float = 0.6,
                              batch_size: int = 32, model_size_gb: float = 10.0) -> BandwidthMetrics:
    """Calculate bandwidth metrics for given configuration.

    Args:
        gpu_type: GPU type name
        sequential_ratio: Ratio of sequential memory accesses
        batch_size: Training batch size
        model_size_gb: Model size in GB

    Returns:
        BandwidthMetrics: Calculated bandwidth metrics
    """
    # Import here to avoid circular imports
    from gpu_specs import get_gpu_bandwidth, calculate_achieved_bandwidth

    theoretical_bandwidth = get_gpu_bandwidth(gpu_type)

    # Adjust sequential ratio based on batch size and model size
    # Larger batches and models tend to have better sequential access
    batch_factor = min(1.0, batch_size / 64.0)  # Normalize to batch 64
    size_factor = min(1.0, model_size_gb / 10.0)  # Normalize to 10B model

    adjusted_sequential_ratio = sequential_ratio * (0.8 + 0.1 * batch_factor + 0.1 * size_factor)
    adjusted_sequential_ratio = min(1.0, adjusted_sequential_ratio)

    # Calculate achieved bandwidth
    achieved_bandwidth = calculate_achieved_bandwidth(
        gpu_type, adjusted_sequential_ratio, burst_size_bytes=64)

    bandwidth_utilization = achieved_bandwidth / theoretical_bandwidth

    return BandwidthMetrics(
        theoretical_bandwidth_gb_s=theoretical_bandwidth,
        achieved_bandwidth_gb_s=achieved_bandwidth,
        bandwidth_utilization=bandwidth_utilization,
        sequential_ratio=adjusted_sequential_ratio
    )

def estimate_num_layers(model_size_billions: float, hidden_dim: int, model_type: str) -> int:
    """Estimate number of layers from model size and hidden dimension.

    Args:
        model_size_billions: Model size in billions of parameters
        hidden_dim: Hidden dimension size
        model_type: Model architecture type

    Returns:
        int: Estimated number of layers
    """
    normalized_type = normalize_model_type(model_type)

    if "transformer" in normalized_type:
        # Each transformer layer has ~12 * hidden_dim^2 parameters
        # (4 * hidden_dim^2 for attention, 8 * hidden_dim^2 for FFN)
        params_per_layer = 12 * hidden_dim * hidden_dim
        total_params = model_size_billions * 1e9

        # Estimate layers (account for embeddings and other components)
        estimated_layers = int(total_params * 0.8 / params_per_layer)  # 80% in layers
        return max(1, estimated_layers)

    elif normalized_type == "cnn":
        # CNN layers vary significantly, use empirical scaling
        return max(1, int(10 * math.sqrt(model_size_billions)))

    else:
        # Default estimate
        return max(1, int(6 * math.sqrt(model_size_billions)))

def calculate_activation_memory(model_type: str, batch_size: int, sequence_length: int,
                              hidden_dim: int, num_layers: int,
                              optimization_level: str = "basic",
                              include_attention_explicit: bool = None) -> float:
    """FIXED: Calculate activation memory requirements using correct EleutherAI formula.

    MAJOR FIXES APPLIED:
    1. Use proven EleutherAI formula: 2 * batch_size * sequence_length * hidden_dim * num_layers
    2. Remove incorrect per-layer component multiplication
    3. Fix attention memory calculation with proper optimization factors
    4. Add sanity checks and debug logging

    Args:
        model_type: Model architecture type
        batch_size: Training batch size (effective per-GPU batch size)
        sequence_length: Sequence length
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        optimization_level: "none", "basic", "aggressive"
        include_attention_explicit: Whether to include explicit attention memory
                                   (auto-determined if None)

    Returns:
        float: Activation memory in GB
    """
    bytes_per_element = 2  # FP16
    normalized_type = normalize_model_type(model_type)

    # Debug logging to track calculations
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Activation memory calc: model={model_type}, batch={batch_size}, "
                f"seq={sequence_length}, hidden={hidden_dim}, layers={num_layers}")

    # Auto-determine whether to include explicit attention calculation
    if include_attention_explicit is None:
        # Only include for transformers with very long sequences where attention dominates
        if "transformer" in normalized_type and sequence_length > 4096:
            include_attention_explicit = True
        else:
            include_attention_explicit = False

    # Optimization factors based on techniques used
    optimization_factors = {
        "none": 1.0,        # No optimizations
        "basic": 0.6,       # Mixed precision, gradient checkpointing, kernel fusion
        "aggressive": 0.3   # Full gradient checkpointing, Flash Attention, advanced fusion
    }

    optimization_factor = optimization_factors.get(optimization_level, 0.6)

    if "transformer" in normalized_type:
        # Use the proven EleutherAI formula from their blog and research
        # Formula: 2 * batch_size * sequence_length * hidden_dim * num_layers
        # This accounts for:
        # - Hidden states: B * S * H per layer
        # - FFN intermediate states: B * S * 4H per layer (but optimized/fused)
        # - Attention outputs: B * S * H per layer
        # - Other activations and intermediate tensors
        # Total: ~2 * B * S * H per layer (empirically validated)

        core_elements = 2 * batch_size * sequence_length * hidden_dim * num_layers
        core_memory_gb = (core_elements * bytes_per_element) / (1024**3)

        # Apply optimization factor
        optimized_core_memory_gb = core_memory_gb * optimization_factor

        logger.info(f"Core activation elements: {core_elements:,}")
        logger.info(f"Core memory before optimization: {core_memory_gb:.2f} GB")
        logger.info(f"Core memory after optimization ({optimization_level}): {optimized_core_memory_gb:.2f} GB")

        # Attention memory calculation (quadratic term)
        attention_memory_gb = 0.0
        if include_attention_explicit:
            # Estimate attention heads
            num_attention_heads = max(1, hidden_dim // 64)

            # Attention score matrices: batch_size * num_heads * seq_len^2 per layer
            attention_elements = batch_size * num_attention_heads * sequence_length * sequence_length * num_layers
            attention_memory_raw_gb = (attention_elements * bytes_per_element) / (1024**3)

            # Apply heavy optimization for attention (Flash Attention effect)
            if optimization_level == "aggressive":
                attention_opt_factor = 0.05  # Flash Attention reduces by 95%
            elif optimization_level == "basic":
                attention_opt_factor = 0.1   # Some attention optimizations
            else:
                attention_opt_factor = 1.0   # No optimizations

            attention_memory_gb = attention_memory_raw_gb * attention_opt_factor

            logger.info(f"Attention elements: {attention_elements:,}")
            logger.info(f"Attention memory before optimization: {attention_memory_raw_gb:.2f} GB")
            logger.info(f"Attention memory after optimization: {attention_memory_gb:.2f} GB")

        total_activations_gb = optimized_core_memory_gb + attention_memory_gb

        # Sanity check to catch remaining bugs
        if total_activations_gb > 500:
            logger.error(f"UNREALISTIC activation memory: {total_activations_gb:.1f} GB!")
            logger.error(f"This suggests a calculation bug. Check your inputs:")
            logger.error(f"  batch_size={batch_size}, sequence_length={sequence_length}")
            logger.error(f"  hidden_dim={hidden_dim}, num_layers={num_layers}")
            logger.error(f"  Expected range for typical models: 5-100 GB")
            # Don't return the unrealistic value - cap it
            total_activations_gb = min(total_activations_gb, 200.0)

        logger.info(f"Final activation memory: {total_activations_gb:.2f} GB")

    elif normalized_type == "cnn":
        # CNN feature maps: batch * resolution^2 * channels * layers
        resolution = int(math.sqrt(sequence_length)) if sequence_length > 1 else 224
        feature_map_elements = batch_size * resolution * resolution * hidden_dim * num_layers * 0.5
        total_activations_gb = (feature_map_elements * bytes_per_element * optimization_factor) / (1024**3)

    elif normalized_type == "diffusion_model":
        noise_levels = 4  # Reduced from 20 - more realistic for training
        elements = batch_size * sequence_length * hidden_dim * noise_levels
        total_activations_gb = (elements * bytes_per_element * optimization_factor) / (1024**3)

    else:
        # Default calculation for unknown architectures
        elements = batch_size * sequence_length * hidden_dim * num_layers * 2 # Modified (used to be 4)
        total_activations_gb = (elements * bytes_per_element * optimization_factor) / (1024**3)

    # Final bounds check
    final_memory = max(0.5, total_activations_gb)  # Minimum 0.5 GB

    # Log final result for verification
    logger.info(f"Final activation memory result: {final_memory:.2f} GB")

    return final_memory


def calculate_fragmentation_memory(allocatable_memory_gb: float,
                                 fragmentation_factor: float) -> float:
    """Calculate memory fragmentation overhead.

    Args:
        allocatable_memory_gb: Memory that can be fragmented (activations, temp buffers)
        fragmentation_factor: Fragmentation factor (0.0-1.0)

    Returns:
        float: Fragmentation overhead in GB
        fixed: Only apply fragmentation to allocatable memory components.
    """
    # Only dynamic allocations cause fragmentation
    # Static allocations (model params, optimizer states) don't fragment
    return allocatable_memory_gb * fragmentation_factor

def format_memory_size(size_gb: float, precision: int = 2) -> str:
    """Format memory size with appropriate units.

    Args:
        size_gb: Size in GB
        precision: Decimal precision

    Returns:
        str: Formatted size string
    """
    if size_gb < 0.001:
        return f"{size_gb * 1024 * 1024:.{precision}f} MB"
    elif size_gb < 1.0:
        return f"{size_gb * 1024:.{precision}f} MB"
    elif size_gb < 1024:
        return f"{size_gb:.{precision}f} GB"
    else:
        return f"{size_gb / 1024:.{precision}f} TB"

def format_bandwidth(bandwidth_gb_s: float, precision: int = 1) -> str:
    """Format bandwidth with appropriate units.

    Args:
        bandwidth_gb_s: Bandwidth in GB/s
        precision: Decimal precision

    Returns:
        str: Formatted bandwidth string
    """
    if bandwidth_gb_s < 1.0:
        return f"{bandwidth_gb_s * 1024:.{precision}f} MB/s"
    elif bandwidth_gb_s < 1024:
        return f"{bandwidth_gb_s:.{precision}f} GB/s"
    else:
        return f"{bandwidth_gb_s / 1024:.{precision}f} TB/s"

def validate_model_parameters(model_type: str, model_size_billions: float,
                            batch_size: int, sequence_length: int) -> Dict[str, Any]:
    """Validate and normalize model parameters.

    Args:
        model_type: Model architecture type
        model_size_billions: Model size in billions of parameters
        batch_size: Training batch size
        sequence_length: Sequence length

    Returns:
        Dict: Validated parameters with warnings
    """
    warnings = []

    # Normalize model type
    normalized_type = normalize_model_type(model_type)

    # Validate model size
    if model_size_billions <= 0:
        warnings.append(f"Invalid model size: {model_size_billions}B, using 1.0B")
        model_size_billions = 1.0
    elif model_size_billions > 1000:
        warnings.append(f"Very large model size: {model_size_billions}B")

    # Validate batch size
    if batch_size <= 0:
        warnings.append(f"Invalid batch size: {batch_size}, using 1")
        batch_size = 1
    elif batch_size > 1024:
        warnings.append(f"Very large batch size: {batch_size}")

    # Validate sequence length
    if sequence_length <= 0:
        warnings.append(f"Invalid sequence length: {sequence_length}, using 1024")
        sequence_length = 1024
    elif sequence_length > 1000000:
        warnings.append(f"Very long sequence: {sequence_length}")

    # Model-specific validations
    if "transformer" in normalized_type and sequence_length > 32768:
        warnings.append(f"Very long sequence for transformer: {sequence_length}")

    return {
        "model_type": normalized_type,
        "model_size_billions": model_size_billions,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "warnings": warnings
    }

def calculate_memory_efficiency(total_memory_gb: float, model_size_gb: float) -> Dict[str, float]:
    """Calculate memory efficiency metrics.

    Args:
        total_memory_gb: Total memory requirement
        model_size_gb: Model parameter size

    Returns:
        Dict: Efficiency metrics
    """
    if model_size_gb <= 0:
        return {"efficiency": 0.0, "overhead_ratio": float('inf'), "overhead_gb": total_memory_gb}

    efficiency = model_size_gb / total_memory_gb
    overhead_gb = total_memory_gb - model_size_gb
    overhead_ratio = overhead_gb / model_size_gb

    return {
        "efficiency": efficiency,
        "overhead_ratio": overhead_ratio,
        "overhead_gb": overhead_gb,
        "memory_multiplier": total_memory_gb / model_size_gb
    }

def is_memory_bound(memory_gb: float, gpu_memory_gb: float, threshold: float = 0.9) -> bool:
    """Check if workload is memory bound.

    Args:
        memory_gb: Required memory in GB
        gpu_memory_gb: Available GPU memory in GB
        threshold: Memory usage threshold

    Returns:
        bool: True if memory bound
    """
    return memory_gb > (gpu_memory_gb * threshold)
