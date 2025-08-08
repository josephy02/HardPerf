import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from utils import normalize_model_type

logger = logging.getLogger(__name__)

@dataclass
class TrainingEstimate:
    """Training time estimation results."""
    total_hours: float
    steps_to_convergence: int
    throughput_tokens_per_sec: float

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "total_hours": self.total_hours,
            "total_days": self.total_hours / 24,
            "steps_to_convergence": self.steps_to_convergence,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec
        }

class TrainingEstimator:
    """Simple training time estimator based on benchmarks and heuristics."""

    # Throughput benchmarks: (model_type, gpu_type) -> {model_size_billions: tokens_per_sec_per_gpu}
    THROUGHPUT_BENCHMARKS = {
        ("transformer_decoder", "NVIDIA H100"): {
            1: 28000,   # 1B model: tokens/sec per GPU
            3: 20000,   # 3B model
            7: 15000,   # 7B model
            13: 12000,  # 13B model
            30: 8000,   # 30B model
            70: 3000,   # 70B model (with tensor parallelism)
            175: 1200,  # 175B model (with advanced parallelism)
        },
        ("transformer_decoder", "NVIDIA H200"): {
            1: 35000,
            3: 25000,
            7: 18000,
            13: 15000,
            30: 10000,
            70: 4000,
            175: 1500,
        },
        ("transformer_decoder", "NVIDIA B200"): {
            1: 45000,
            3: 32000,
            7: 24000,
            13: 20000,
            30: 14000,
            70: 6000,
            175: 2200,
        },
        ("transformer_decoder", "NVIDIA A100"): {
            1: 20000,
            3: 15000,
            7: 12000,
            13: 9000,
            30: 6000,
            70: 2200,
            175: 800,
        },
        ("transformer_decoder", "AMD MI300X"): {
            1: 22000,
            3: 16000,
            7: 13000,
            13: 10000,
            30: 6500,
            70: 2500,
            175: 900,
        },
        ("vision_transformer", "NVIDIA H100"): {
            1: 35000,   # ViTs are generally more efficient
            3: 28000,
            7: 22000,
            13: 18000,
            30: 12000,
            70: 8000,
        },
        ("vision_transformer", "NVIDIA H200"): {
            1: 42000,
            3: 35000,
            7: 28000,
            13: 22000,
            30: 15000,
            70: 10000,
        },
        ("vision_transformer", "NVIDIA B200"): {
            1: 55000,
            3: 45000,
            7: 38000,
            13: 30000,
            30: 20000,
            70: 14000,
        },
        ("cnn", "NVIDIA H100"): {
            0.1: 45000,  # CNNs typically smaller but more efficient
            0.5: 38000,
            1: 32000,
            3: 25000,
            7: 18000,
        },
        ("cnn", "NVIDIA H200"): {
            0.1: 55000,
            0.5: 48000,
            1: 42000,
            3: 32000,
            7: 24000,
        },
        ("diffusion_model", "NVIDIA H100"): {
            1: 12000,   # Diffusion models are more compute intensive
            3: 9000,
            7: 6500,
            13: 4500,
            30: 2800,
        },
        ("diffusion_model", "NVIDIA H200"): {
            1: 15000,
            3: 11000,
            7: 8000,
            13: 5500,
            30: 3500,
        },
    }

    # Steps needed for convergence heuristics
    CONVERGENCE_STEPS = {
        "transformer_decoder": {
            "base_steps_per_billion": 1000,  # 1000 steps per billion parameters
            "sequence_length_factor": 1.0,   # Longer sequences don't need more steps
            "batch_size_factor": 0.8,        # Smaller batches need more steps
        },
        "vision_transformer": {
            "base_steps_per_billion": 800,   # ViTs converge faster
            "sequence_length_factor": 0.5,   # Less sensitive to sequence length
            "batch_size_factor": 0.7,
        },
        "cnn": {
            "base_steps_per_billion": 1500,  # CNNs need more steps due to smaller size
            "sequence_length_factor": 0.3,
            "batch_size_factor": 0.9,
        },
        "diffusion_model": {
            "base_steps_per_billion": 2000,  # Diffusion models need many steps
            "sequence_length_factor": 0.2,
            "batch_size_factor": 0.6,
        }
    }

    def __init__(self):
        """Initialize training estimator."""
        pass

    def estimate_training_time(self, model_info: Dict, cluster_spec) -> TrainingEstimate:
        """Estimate training time for given model and hardware configuration.

        Args:
            model_info: Dictionary containing model configuration
            cluster_spec: Cluster specification with GPU info

        Returns:
            TrainingEstimate: Time estimation results
        """
        model_type = model_info.get("model_type", "transformer_decoder")
        model_size = model_info.get("model_size_billions", 1.0)
        batch_size = model_info.get("batch_size", 32)
        sequence_length = model_info.get("sequence_length", 2048)

        # Get throughput estimate
        throughput_per_gpu = self._get_throughput_estimate(model_type, model_size, cluster_spec.gpu_type)

        # Calculate effective cluster throughput
        effective_throughput = self._calculate_effective_throughput(
            throughput_per_gpu, cluster_spec, model_info
        )

        # Estimate steps needed for convergence
        steps_needed = self._estimate_convergence_steps(model_type, model_size, batch_size, sequence_length)

        # Calculate training time
        tokens_per_step = batch_size * sequence_length
        seconds_per_step = tokens_per_step / effective_throughput
        total_hours = (steps_needed * seconds_per_step) / 3600

        logger.info(f"Training estimate: {total_hours:.1f} hours for {steps_needed:,} steps "
                   f"at {effective_throughput:,.0f} tokens/sec")

        return TrainingEstimate(
            total_hours=total_hours,
            steps_to_convergence=steps_needed,
            throughput_tokens_per_sec=effective_throughput
        )

    def _get_throughput_estimate(self, model_type: str, model_size: float, gpu_type: str) -> float:
        """Get throughput estimate for single GPU."""
        # Normalize model type using utils function
        model_key = normalize_model_type(model_type)

        # Get benchmark data
        key = (model_key, gpu_type)
        if key not in self.THROUGHPUT_BENCHMARKS:
            # Fallback to H100 if GPU type not found
            fallback_key = (model_key, "NVIDIA H100")
            if fallback_key in self.THROUGHPUT_BENCHMARKS:
                logger.warning(f"No benchmarks for {gpu_type}, using H100 estimates")
                key = fallback_key
            else:
                # Last resort fallback
                logger.warning(f"No benchmarks for {model_type} on {gpu_type}, using default estimates")
                return 10000.0  # Conservative default

        benchmarks = self.THROUGHPUT_BENCHMARKS[key]

        # Find closest model size in benchmarks
        sizes = sorted(benchmarks.keys())
        closest_size = min(sizes, key=lambda x: abs(x - model_size))

        # Interpolate if model size is between two benchmarks
        if model_size != closest_size and len(sizes) > 1:
            throughput = self._interpolate_throughput(model_size, benchmarks)
        else:
            throughput = benchmarks[closest_size]

        return throughput

    def _interpolate_throughput(self, model_size: float, benchmarks: Dict[float, float]) -> float:
        """Interpolate throughput for model sizes between benchmarks."""
        sizes = sorted(benchmarks.keys())

        # Find surrounding sizes
        lower = max([s for s in sizes if s <= model_size], default=sizes[0])
        upper = min([s for s in sizes if s >= model_size], default=sizes[-1])

        if lower == upper:
            return benchmarks[lower]

        # Linear interpolation
        weight = (model_size - lower) / (upper - lower)
        throughput = benchmarks[lower] * (1 - weight) + benchmarks[upper] * weight

        return throughput

    def _calculate_effective_throughput(self, throughput_per_gpu: float, cluster_spec, model_info: Dict) -> float:
        """Calculate effective cluster throughput considering parallelization overhead."""
        if cluster_spec.num_gpus == 1:
            return throughput_per_gpu

        # Base cluster throughput
        base_throughput = throughput_per_gpu * cluster_spec.num_gpus

        # Apply parallelization efficiency based on strategy
        is_multi_gpu = cluster_spec.num_gpus > 1
        if not is_multi_gpu:
            return base_throughput

        # Estimate parallelization efficiency based on typical overheads
        if cluster_spec.num_gpus <= 8:
            # Single node - good efficiency
            if cluster_spec.interconnect.startswith("nvlink"):
                efficiency = 0.90  # NVLink is more efficient
            else:
                efficiency = 0.80  # Other interconnects
        else:
            # Multi-node - more overhead
            if cluster_spec.interconnect.startswith("nvswitch"):
                efficiency = 0.85
            elif "infiniband" in cluster_spec.interconnect:
                efficiency = 0.70
            else:
                efficiency = 0.60

        # Additional penalty for tensor parallelism (high communication)
        model_size = model_info.get("model_size_billions", 1.0)
        if model_size > 30 and cluster_spec.num_gpus > 4:
            # Large models likely use tensor parallelism
            efficiency *= 0.85

        return base_throughput * efficiency

    def _estimate_convergence_steps(self, model_type: str, model_size: float,
                                   batch_size: int, sequence_length: int) -> int:
        """Estimate number of steps needed for convergence."""
        # Normalize model type using utils function
        model_key = normalize_model_type(model_type)

        if model_key not in self.CONVERGENCE_STEPS:
            model_key = "transformer_decoder"  # Default

        config = self.CONVERGENCE_STEPS[model_key]

        # Base steps
        base_steps = config["base_steps_per_billion"] * model_size

        # Adjust for batch size (smaller batches need more steps)
        batch_factor = max(0.5, min(2.0, 32.0 / batch_size))  # Normalize to batch 32
        batch_adjustment = 1.0 + (batch_factor - 1.0) * config["batch_size_factor"]

        # Adjust for sequence length (longer sequences may need slightly more steps)
        seq_factor = max(0.8, min(1.5, sequence_length / 2048.0))  # Normalize to 2048
        seq_adjustment = 1.0 + (seq_factor - 1.0) * config["sequence_length_factor"]

        steps = int(base_steps * batch_adjustment * seq_adjustment)

        # Reasonable bounds
        steps = max(1000, min(1000000, steps))

        return steps

    def get_throughput_info(self, model_type: str, gpu_type: str) -> Optional[Dict]:
        """Get available throughput benchmark information."""
        # Normalize model type using utils function
        model_key = normalize_model_type(model_type)
        key = (model_key, gpu_type)

        if key in self.THROUGHPUT_BENCHMARKS:
            return {
                "model_type": model_key,
                "gpu_type": gpu_type,
                "benchmarks": self.THROUGHPUT_BENCHMARKS[key]
            }
        return None
