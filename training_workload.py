import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from utils import (
    ModelArchitecture, get_memory_constants, estimate_hidden_dimension,
    estimate_num_layers, calculate_activation_memory, BandwidthMetrics,
    calculate_bandwidth_metrics, validate_model_parameters, calculate_fragmentation_memory,
    normalize_model_type
)
from gpu_specs import get_gpu_memory_capacity

logger = logging.getLogger(__name__)

@dataclass
class MemoryBreakdown:
    """Memory usage breakdown by component."""
    model_parameters_gb: float
    optimizer_states_gb: float
    gradients_gb: float
    activations_gb: float
    temporary_buffers_gb: float
    framework_overhead_gb: float
    fragmentation_gb: float
    total_memory_gb: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "model_parameters_gb": self.model_parameters_gb,
            "optimizer_states_gb": self.optimizer_states_gb,
            "gradients_gb": self.gradients_gb,
            "activations_gb": self.activations_gb,
            "temporary_buffers_gb": self.temporary_buffers_gb,
            "framework_overhead_gb": self.framework_overhead_gb,
            "fragmentation_gb": self.fragmentation_gb,
            "total_memory_gb": self.total_memory_gb
        }

@dataclass
class AccessPatterns:
    """Memory access pattern characteristics."""
    spatial_locality: float       # 0.0-1.0, higher is better
    temporal_locality: float      # 0.0-1.0, higher is better
    sequential_ratio: float       # 0.0-1.0, fraction of sequential accesses
    working_set_size_mb: float    # Working set size in MB
    bandwidth_utilization: float  # 0.0-1.0, achieved/theoretical bandwidth

    def to_dict(self) -> Dict[str, float]:
        return {
            "spatial_locality": self.spatial_locality,
            "temporal_locality": self.temporal_locality,
            "sequential_ratio": self.sequential_ratio,
            "working_set_size_mb": self.working_set_size_mb,
            "bandwidth_utilization": self.bandwidth_utilization
        }

class BaseWorkloadModel(ABC):
    """Base class for workload models."""

    def __init__(self, model_size_billions: float, batch_size: int,
                 sequence_length: int, hidden_dim: Optional[int] = None,
                 gpu_type: str = "NVIDIA H100", effective_batch_size: Optional[int] = None):
        self.model_size_billions = model_size_billions
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.gpu_type = gpu_type

        # Support for effective batch size (for multi-GPU)
        # effective_batch_size is the per-GPU batch size in multi-GPU setups
        self.effective_batch_size = effective_batch_size or batch_size

        # Don't call get_model_type() here - let child classes call _initialize_derived_attributes()
        self.hidden_dim = hidden_dim
        self.num_layers = None  # Will be set by child class

    def _initialize_derived_attributes(self):
        """Initialize derived attributes after child class attributes are set."""
        # Estimate hidden dim if not provided
        if self.hidden_dim is None:
            self.hidden_dim = estimate_hidden_dimension(self.model_size_billions, self.get_model_type())

        # Estimate number of layers
        self.num_layers = estimate_num_layers(
            self.model_size_billions, self.hidden_dim, self.get_model_type())

    @abstractmethod
    def get_model_type(self) -> str:
        """Get model architecture type string."""
        pass

    @abstractmethod
    def calculate_memory_breakdown(self) -> MemoryBreakdown:
        """Calculate detailed memory breakdown."""
        pass

    @abstractmethod
    def analyze_access_patterns(self) -> AccessPatterns:
        """Analyze memory access patterns."""
        pass

    def get_basic_memory_breakdown(self) -> MemoryBreakdown:
        """Calculate basic memory breakdown using constants."""
        constants = get_memory_constants(self.get_model_type())

        # Base components
        model_params_gb = self.model_size_billions
        optimizer_states_gb = model_params_gb * constants["optimizer_factor"]
        gradients_gb = model_params_gb

        # Use effective batch size for activation calculation
        activations_gb = calculate_activation_memory(
            self.get_model_type(), self.effective_batch_size, self.sequence_length,
            self.hidden_dim, self.num_layers
        )

        # Derived components
        temp_buffers_gb = activations_gb * constants["temp_buffer_factor"]
        framework_overhead_gb = (constants["framework_overhead_base"] +
                               model_params_gb * constants["framework_overhead_factor"])

        # Only apply fragmentation to allocatable memory
        allocatable_memory_gb = activations_gb + temp_buffers_gb
        fragmentation_gb = calculate_fragmentation_memory(
            allocatable_memory_gb, constants["fragmentation_factor"]
        )

        total_memory_gb = (model_params_gb + optimizer_states_gb + gradients_gb +
                          activations_gb + temp_buffers_gb + framework_overhead_gb +
                          fragmentation_gb)

        return MemoryBreakdown(
            model_parameters_gb=model_params_gb,
            optimizer_states_gb=optimizer_states_gb,
            gradients_gb=gradients_gb,
            activations_gb=activations_gb,
            temporary_buffers_gb=temp_buffers_gb,
            framework_overhead_gb=framework_overhead_gb,
            fragmentation_gb=fragmentation_gb,
            total_memory_gb=total_memory_gb
        )

class TransformerModel(BaseWorkloadModel):
    """Transformer architecture workload model."""

    def __init__(self, model_size_billions: float, batch_size: int,
                 sequence_length: int, hidden_dim: Optional[int] = None,
                 is_decoder: bool = True, num_attention_heads: Optional[int] = None,
                 gpu_type: str = "NVIDIA H100", effective_batch_size: Optional[int] = None):
        # Set transformer-specific attributes first
        self.is_decoder = is_decoder

        # Call parent constructor
        super().__init__(model_size_billions, batch_size, sequence_length,
                        hidden_dim, gpu_type, effective_batch_size)

        # Now initialize derived attributes after all attributes are set
        self._initialize_derived_attributes()

        # Set attention heads
        self.num_attention_heads = num_attention_heads or (self.hidden_dim // 64)

    def get_model_type(self) -> str:
        return "transformer_decoder" if self.is_decoder else "transformer_encoder"

    def calculate_memory_breakdown(self) -> MemoryBreakdown:
        """Calculate transformer-specific memory breakdown."""
        # Start with basic breakdown
        breakdown = self.get_basic_memory_breakdown()

        # Transformer-specific adjustments
        if self.sequence_length > 4096:
            # Long sequences increase attention memory pressure
            attention_scaling = min(2.0, self.sequence_length / 4096)
            breakdown.activations_gb *= (1.0 + 0.3 * (attention_scaling - 1.0))

        if self.is_decoder:
            # Decoder models need KV cache for inference
            kv_cache_gb = (self.effective_batch_size * self.sequence_length *
                           self.hidden_dim * 2 * 2) / (1024**3)  # 2 tensors, FP16
            breakdown.activations_gb += kv_cache_gb * 0.2  # Partial allocation during training

        # Recalculate fragmentation with new activation size
        constants = get_memory_constants(self.get_model_type())
        allocatable_memory_gb = breakdown.activations_gb + breakdown.temporary_buffers_gb
        breakdown.fragmentation_gb = calculate_fragmentation_memory(
            allocatable_memory_gb, constants["fragmentation_factor"]
        )

        # Recalculate total
        breakdown.total_memory_gb = (
            breakdown.model_parameters_gb + breakdown.optimizer_states_gb +
            breakdown.gradients_gb + breakdown.activations_gb +
            breakdown.temporary_buffers_gb + breakdown.framework_overhead_gb +
            breakdown.fragmentation_gb
        )

        return breakdown

    def analyze_access_patterns(self) -> AccessPatterns:
        """Analyze transformer access patterns."""
        # Transformers have mixed access patterns
        # - Sequential for matrix multiplications
        # - Random for attention mechanisms

        # Base patterns
        if self.sequence_length <= 2048:
            spatial_locality = 0.7
            temporal_locality = 0.6
            sequential_ratio = 0.6
        else:
            # Longer sequences reduce locality
            spatial_locality = 0.6
            temporal_locality = 0.5
            sequential_ratio = 0.5

        # Use effective batch size for working set calculation
        working_set_size_mb = (self.model_size_billions * 1024 * 0.3 +  # 30% of model
                               self.effective_batch_size * self.sequence_length * self.hidden_dim * 2 / (1024**2))

        # Calculate bandwidth metrics
        bandwidth_metrics = calculate_bandwidth_metrics(self.gpu_type, sequential_ratio,
                                                       self.effective_batch_size, self.model_size_billions)

        return AccessPatterns(
            spatial_locality=spatial_locality,
            temporal_locality=temporal_locality,
            sequential_ratio=sequential_ratio,
            working_set_size_mb=working_set_size_mb,
            bandwidth_utilization=bandwidth_metrics.bandwidth_utilization
        )

class CNNModel(BaseWorkloadModel):
    """CNN architecture workload model."""

    def __init__(self, model_size_billions: float, batch_size: int,
                 sequence_length: int, hidden_dim: Optional[int] = None,
                 num_channels: int = 3, gpu_type: str = "NVIDIA H100",
                 effective_batch_size: Optional[int] = None):
        # Set CNN-specific attributes first
        self.num_channels = num_channels
        self.image_resolution = int(math.sqrt(sequence_length)) if sequence_length > 1 else 224

        # Call parent constructor
        super().__init__(model_size_billions, batch_size, sequence_length,
                        hidden_dim, gpu_type, effective_batch_size)

        # Initialize derived attributes
        self._initialize_derived_attributes()

    def get_model_type(self) -> str:
        return "cnn"

    def calculate_memory_breakdown(self) -> MemoryBreakdown:
        """Calculate CNN-specific memory breakdown."""
        # Start with basic breakdown
        breakdown = self.get_basic_memory_breakdown()

        # CNN-specific adjustments
        # Feature maps scale quadratically with resolution
        if self.image_resolution > 512:
            resolution_scaling = (self.image_resolution / 512) ** 2
            breakdown.activations_gb *= (1.0 + 0.5 * (resolution_scaling - 1.0))

        # CNNs typically use less optimizer memory
        breakdown.optimizer_states_gb *= 0.8  # Often use SGD or lighter optimizers

        # Recalculate fragmentation with new activation size
        constants = get_memory_constants(self.get_model_type())
        allocatable_memory_gb = breakdown.activations_gb + breakdown.temporary_buffers_gb
        breakdown.fragmentation_gb = calculate_fragmentation_memory(
            allocatable_memory_gb, constants["fragmentation_factor"]
        )

        # Recalculate total
        breakdown.total_memory_gb = (
            breakdown.model_parameters_gb + breakdown.optimizer_states_gb +
            breakdown.gradients_gb + breakdown.activations_gb +
            breakdown.temporary_buffers_gb + breakdown.framework_overhead_gb +
            breakdown.fragmentation_gb
        )

        return breakdown

    def analyze_access_patterns(self) -> AccessPatterns:
        """Analyze CNN access patterns."""
        # CNNs have excellent spatial locality due to convolution operations
        spatial_locality = 0.85  # Convolutions access neighboring pixels
        temporal_locality = 0.75  # Filter weights are reused across positions
        sequential_ratio = 0.8    # Mostly sequential access through feature maps

        # Use effective batch size and realistic working set
        feature_map_mb = (self.effective_batch_size * self.image_resolution**2 *
                          self.hidden_dim) / (1024**2)
        filter_mb = (self.model_size_billions * 1024 * 0.2)  # 20% of params
        working_set_size_mb = feature_map_mb + filter_mb

        # Calculate bandwidth metrics
        bandwidth_metrics = calculate_bandwidth_metrics(self.gpu_type, sequential_ratio,
                                                       self.effective_batch_size, self.model_size_billions)

        return AccessPatterns(
            spatial_locality=spatial_locality,
            temporal_locality=temporal_locality,
            sequential_ratio=sequential_ratio,
            working_set_size_mb=working_set_size_mb,
            bandwidth_utilization=bandwidth_metrics.bandwidth_utilization
        )

class DiffusionModel(BaseWorkloadModel):
    """Diffusion model workload model."""

    def __init__(self, model_size_billions: float, batch_size: int,
                 sequence_length: int, hidden_dim: Optional[int] = None,
                 num_timesteps: int = 1000, num_channels: int = 4,
                 gpu_type: str = "NVIDIA H100", effective_batch_size: Optional[int] = None):
        # Set diffusion-specific attributes first
        self.num_timesteps = num_timesteps
        self.num_channels = num_channels
        self.image_resolution = int(math.sqrt(sequence_length)) if sequence_length > 1 else 64

        # Call parent constructor
        super().__init__(model_size_billions, batch_size, sequence_length,
                        hidden_dim, gpu_type, effective_batch_size)

        # Initialize derived attributes
        self._initialize_derived_attributes()

    def get_model_type(self) -> str:
        return "diffusion_model"

    def calculate_memory_breakdown(self) -> MemoryBreakdown:
        """Calculate diffusion-specific memory breakdown."""
        # Start with basic breakdown
        breakdown = self.get_basic_memory_breakdown()

        # Diffusion-specific adjustments
        # Multiple timesteps increase activation memory significantly
        timestep_factor = min(3.0, math.sqrt(self.num_timesteps / 20))  # Normalize to 20 steps
        breakdown.activations_gb *= timestep_factor

        # U-Net architecture has skip connections increasing memory
        breakdown.activations_gb *= 1.3

        # Additional noise scheduling buffers
        noise_buffers_gb = (self.effective_batch_size * self.image_resolution**2 *
                           self.num_channels * 4) / (1024**3)  # Multiple noise levels
        breakdown.temporary_buffers_gb += noise_buffers_gb

        # Recalculate fragmentation with new activation and buffer sizes
        constants = get_memory_constants(self.get_model_type())
        allocatable_memory_gb = breakdown.activations_gb + breakdown.temporary_buffers_gb
        breakdown.fragmentation_gb = calculate_fragmentation_memory(
            allocatable_memory_gb, constants["fragmentation_factor"]
        )

        # Recalculate total
        breakdown.total_memory_gb = (
            breakdown.model_parameters_gb + breakdown.optimizer_states_gb +
            breakdown.gradients_gb + breakdown.activations_gb +
            breakdown.temporary_buffers_gb + breakdown.framework_overhead_gb +
            breakdown.fragmentation_gb
        )

        return breakdown

    def analyze_access_patterns(self) -> AccessPatterns:
        """Analyze diffusion model access patterns."""
        # Diffusion models have mixed patterns due to U-Net architecture
        spatial_locality = 0.65   # Skip connections reduce locality
        temporal_locality = 0.45  # Different timesteps reduce reuse
        sequential_ratio = 0.55   # Mix of convolution and attention-like operations

        # Use effective batch size and realistic working set
        multi_resolution_mb = (self.effective_batch_size * self.image_resolution**2 *
                               self.hidden_dim * 2) / (1024**2)  # Encoder + decoder
        timestep_mb = (self.num_timesteps * self.hidden_dim * 4) / (1024**2)  # Timestep embeddings
        working_set_size_mb = multi_resolution_mb + timestep_mb

        # Calculate bandwidth metrics
        bandwidth_metrics = calculate_bandwidth_metrics(self.gpu_type, sequential_ratio,
                                                       self.effective_batch_size, self.model_size_billions)

        return AccessPatterns(
            spatial_locality=spatial_locality,
            temporal_locality=temporal_locality,
            sequential_ratio=sequential_ratio,
            working_set_size_mb=working_set_size_mb,
            bandwidth_utilization=bandwidth_metrics.bandwidth_utilization
        )

class VisionTransformerModel(BaseWorkloadModel):
    """Vision Transformer workload model."""

    def __init__(self, model_size_billions: float, batch_size: int,
                 sequence_length: int, hidden_dim: Optional[int] = None,
                 patch_size: int = 16, num_channels: int = 3,
                 gpu_type: str = "NVIDIA H100", effective_batch_size: Optional[int] = None):
        # Set ViT-specific attributes first
        self.patch_size = patch_size
        self.num_channels = num_channels
        # ViT image resolution calculation is correct
        self.image_resolution = int(math.sqrt(sequence_length)) * patch_size

        # Call parent constructor
        super().__init__(model_size_billions, batch_size, sequence_length,
                        hidden_dim, gpu_type, effective_batch_size)

        # Initialize derived attributes
        self._initialize_derived_attributes()

    def get_model_type(self) -> str:
        return "vision_transformer"

    def calculate_memory_breakdown(self) -> MemoryBreakdown:
        """Calculate ViT-specific memory breakdown."""
        # Start with basic breakdown (similar to transformer)
        breakdown = self.get_basic_memory_breakdown()

        # ViT-specific adjustments
        # Patch embeddings add some overhead
        patch_embedding_gb = (self.effective_batch_size * self.sequence_length *
                              self.hidden_dim * 2) / (1024**3)
        breakdown.activations_gb += patch_embedding_gb * 0.1

        # Position embeddings
        pos_embedding_gb = (self.sequence_length * self.hidden_dim * 2) / (1024**3)
        breakdown.model_parameters_gb += pos_embedding_gb

        # Recalculate fragmentation with new activation size
        constants = get_memory_constants(self.get_model_type())
        allocatable_memory_gb = breakdown.activations_gb + breakdown.temporary_buffers_gb
        breakdown.fragmentation_gb = calculate_fragmentation_memory(
            allocatable_memory_gb, constants["fragmentation_factor"]
        )

        # Recalculate total
        breakdown.total_memory_gb = (
            breakdown.model_parameters_gb + breakdown.optimizer_states_gb +
            breakdown.gradients_gb + breakdown.activations_gb +
            breakdown.temporary_buffers_gb + breakdown.framework_overhead_gb +
            breakdown.fragmentation_gb
        )

        return breakdown

    def analyze_access_patterns(self) -> AccessPatterns:
        """Analyze ViT access patterns."""
        # ViTs combine patch-based processing with transformer attention
        spatial_locality = 0.75   # Patch-based processing has good locality
        temporal_locality = 0.55  # Attention reduces temporal locality
        sequential_ratio = 0.65   # Mix of sequential patch processing and attention

        # Use effective batch size and realistic working set
        patch_mb = (self.effective_batch_size * self.sequence_length *
                    self.hidden_dim * 2) / (1024**2)
        attention_mb = (self.model_size_billions * 1024 * 0.15)  # 15% in attention
        working_set_size_mb = patch_mb + attention_mb

        # Calculate bandwidth metrics
        bandwidth_metrics = calculate_bandwidth_metrics(self.gpu_type, sequential_ratio,
                                                       self.effective_batch_size, self.model_size_billions)

        return AccessPatterns(
            spatial_locality=spatial_locality,
            temporal_locality=temporal_locality,
            sequential_ratio=sequential_ratio,
            working_set_size_mb=working_set_size_mb,
            bandwidth_utilization=bandwidth_metrics.bandwidth_utilization
        )

def create_workload_model(model_type: str, model_size_billions: float,
                         batch_size: int, sequence_length: int,
                         gpu_type: str = "NVIDIA H100",
                         effective_batch_size: Optional[int] = None,
                         **kwargs) -> BaseWorkloadModel:
    """Factory function to create appropriate workload model.

    Args:
        model_type: Model architecture type
        model_size_billions: Model size in billions of parameters
        batch_size: Training batch size
        sequence_length: Sequence length
        gpu_type: GPU type name
        effective_batch_size: Effective per-GPU batch size (for multi-GPU)
        **kwargs: Additional model-specific parameters

    Returns:
        BaseWorkloadModel: Appropriate workload model instance
    """
    # Validate parameters
    validated = validate_model_parameters(model_type, model_size_billions,
                                        batch_size, sequence_length)

    if validated["warnings"]:
        for warning in validated["warnings"]:
            logger.warning(warning)

    # Extract validated parameters
    model_type = validated["model_type"]
    model_size_billions = validated["model_size_billions"]
    batch_size = validated["batch_size"]
    sequence_length = validated["sequence_length"]

    # Normalize model type using utils function
    normalized_type = normalize_model_type(model_type)

    # Create appropriate model
    if "transformer" in normalized_type:
        is_decoder = "decoder" in normalized_type or normalized_type == "transformer"  # Default to decoder
        return TransformerModel(
            model_size_billions, batch_size, sequence_length,
            is_decoder=is_decoder, gpu_type=gpu_type,
            effective_batch_size=effective_batch_size, **kwargs
        )

    elif normalized_type == "cnn":
        return CNNModel(model_size_billions, batch_size, sequence_length,
                       gpu_type=gpu_type, effective_batch_size=effective_batch_size, **kwargs)

    elif normalized_type == "diffusion_model":
        return DiffusionModel(model_size_billions, batch_size, sequence_length,
                             gpu_type=gpu_type, effective_batch_size=effective_batch_size, **kwargs)

    elif normalized_type == "vision_transformer":
        return VisionTransformerModel(model_size_billions, batch_size, sequence_length,
                                     gpu_type=gpu_type, effective_batch_size=effective_batch_size, **kwargs)

    else:
        logger.warning(f"Unknown model type '{model_type}', using TransformerModel")
        return TransformerModel(model_size_billions, batch_size, sequence_length,
                              is_decoder=True, gpu_type=gpu_type,
                              effective_batch_size=effective_batch_size, **kwargs)

def estimate_max_batch_size(model_type: str, model_size_billions: float,
                           sequence_length: int, gpu_type: str = "NVIDIA H100",
                           memory_efficiency: float = 0.9) -> int:
    """Estimate maximum batch size that fits in GPU memory.

    Args:
        model_type: Model architecture type
        model_size_billions: Model size in billions of parameters
        sequence_length: Sequence length
        gpu_type: GPU type name
        memory_efficiency: Memory efficiency factor (0.0-1.0)

    Returns:
        int: Maximum batch size
    """
    gpu_memory_gb = get_gpu_memory_capacity(gpu_type)
    available_memory_gb = gpu_memory_gb * memory_efficiency

    # Binary search for maximum batch size
    min_batch = 1
    max_batch = 2048
    best_batch = 1

    while min_batch <= max_batch:
        test_batch = (min_batch + max_batch) // 2

        # Create model and estimate memory
        model = create_workload_model(model_type, model_size_billions,
                                    test_batch, sequence_length, gpu_type)
        memory_breakdown = model.calculate_memory_breakdown()

        if memory_breakdown.total_memory_gb <= available_memory_gb:
            best_batch = test_batch
            min_batch = test_batch + 1
        else:
            max_batch = test_batch - 1

    return best_batch