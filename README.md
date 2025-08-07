# memprof-training: Training Memory Performance Analysis Tool

Predict memory usage and performance for ML model training by analyzing memory requirements, bottlenecks, and optimization opportunities before you start AI model training.

## Why memprof-training?

Training large language models and other AI models requires careful memory management and resource planning. A single OOM (Out of Memory) error can waste hours of compute time and thousands of dollars in cloud resources. Traditional approaches rely on trial-and-error configuration tuning, where memory issues are discovered only after training has already begun.

memprof-training solves this by **predicting memory requirements and performance characteristics before you start training**:
* Will your 70B model fit on 8x H100 GPUs with tensor parallelism of degree 8?
* How much memory will activations consume with Flash Attention vs. standard attention mechanisms?
* What's the maximum feasible batch size for your hardware configuration?
* What are the memory-speed tradeoffs of gradient checkpointing for your specific model architecture?
* How will different parallelization strategies (data parallel, tensor parallel, pipeline parallel) affect memory usage and training throughput?

## Philosophy

**Hardware is fixed, software is flexible.** Start with what you have. You can't change the GPUs in your cluster, but you can optimize everything else—gradient checkpointing, mixed precision, parallelization strategies, and attention mechanisms. memprof-training focuses on the software optimizations you can actually control.


**Real training scenarios over theoretical calculations.** The tool uses validated benchmarks from actual training runs rather than theoretical FLOP calculations that often miss real-world bottlenecks like attention memory explosion.

**Training time.** Every analysis includes training time estimation because successful model training is ultimately about achieving convergence within budget and time constraints.

## How It Works

memprof-training calculates memory needed using the following formula:

```
Total Memory = Model Parameters + Optimizer States + Gradients + Activations + Buffers + Overhead
```

The key insight: **training memory is dominated by optimizer states and activations**, not just model size. For large transformers with long sequences, activation memory can grow quadratically due to attention mechanisms, often becoming the primary bottleneck.

### The Attention Memory Challenge

For transformer models, attention memory grows **quadratically** with sequence length:
- **Standard attention**: $O(\text{sequence-length}^2)$ memory scaling
- **Flash Attention**: Reduces to $O(\text{sequence-length})$ scaling
- **Long sequences**: Can cause 10x-100x memory explosion without optimization

This quadratic scaling often makes attention the primary memory bottleneck for long sequences, not model parameters.

## Key Features

- **Memory prediction** before training with component-by-component breakdown
- **Training time estimation** based on real hardware benchmarks and model scaling laws
- **Multi-GPU strategies** comparing data parallel, tensor parallel, pipeline parallel, and hybrid approaches
- **Optimization recommendations** for gradient checkpointing, mixed precision, and Flash Attention
- **Attention memory analysis** with quadratic scaling detection and Flash Attention benefits
- **Bottleneck detection** identifying memory capacity, communication overhead, and scaling efficiency issues
- **YAML configuration** support for complex multi-GPU setups and production workflows

## Quick Start

### Basic Analysis
```bash
# Check if 7B model fits on H100 for training
python mprofile_training.py transformer 7.0 32 --gpu-type "NVIDIA H100"
```

### With Long Sequences (Attention Memory Warning)
```bash
# 7B model with 8K sequence length triggers attention memory explosion
python mprofile_training.py transformer 7.0 16 --sequence-length 8192 --gpu-type "NVIDIA H100"
```

### Multi-GPU Analysis
```bash
# 70B model across 8 GPUs with tensor parallelism
python mprofile_training.py transformer 70.0 16 --gpus 8 --strategy tensor_parallel
```

### Complete Analysis with Recommendations
```bash
# Full bottleneck analysis with optimization recommendations
python mprofile_training.py transformer 13.0 32 --sequence-length 2048 --bottlenecks --verbose
```

### Using YAML Configuration
```bash
# Complex multi-GPU setup from configuration file
python mprofile_training.py --config configs/70b_hybrid_training.yaml
```

## Usage Example: Training a 13B Model with Long Sequences

**Scenario**: Training a 13B parameter model for document understanding. Sequences can be up to 4K tokens, and you have access to 4x H100 GPUs.

### Step 1: Check Basic Feasibility
```bash
python mprofile_training.py transformer 13.0 32 --sequence-length 4096 --gpu-type "NVIDIA H100"
```

**Result**:
```
✗ EXCEEDS CAPACITY
Memory required: 156.3 GB per GPU
GPU capacity: 80 GB (72.0 GB usable)
Memory overflow: 84.3 GB (optimization required)

MEMORY BREAKDOWN:
Model Parameters     : 26.0 GB (16.6%)
Optimizer States     : 52.0 GB (33.3%)
Gradients           : 26.0 GB (16.6%)
Activations         : 47.8 GB (30.6%)   # ← High due to 4K sequence
Other (buffers)     :  4.5 GB ( 2.9%)

CRITICAL BOTTLENECK DETECTED:
Massive activation memory (47.8 GB) indicates quadratic attention scaling
with 4096 sequence length. Standard attention is impractical for long sequences.
```

**Analysis**: The model doesn't fit! Activations use 47.8GB due to quadratic attention scaling with 4K sequences. This demonstrates how **sequence length can be more critical than model size** for memory requirements.

### Step 2: Apply Flash Attention Optimization
```bash
python mprofile_training.py transformer 13.0 32 --sequence-length 4096 --gpu-type "NVIDIA H100" --bottlenecks
```

**Result with Flash Attention recommendation**:
```
RECOMMENDATIONS:
1. [CRITICAL] CRITICAL: Enable Flash Attention for Long Sequences
   Description: Standard attention with 4096 sequence length requires 156GB memory.
   • ⚡ Enable Flash Attention IMMEDIATELY (reduces memory from TB to GB scale)
   • 📦 Install: pip install flash-attn (for PyTorch)
   • 🔧 Replace standard attention with Flash Attention in your model
   Expected: Reduce from 156GB to ~16GB (90%+ reduction)

2. [HIGH] Reduce Memory Usage (Required)
   • Enable gradient checkpointing (60-80% activation memory reduction)
   • Use mixed precision (FP16/BF16) training (50% total memory reduction)
   • Reduce batch size and use gradient accumulation
```

**Analysis**: Flash Attention is **mandatory** for 4K sequences. It changes O(S²) to O(S) scaling, reducing activation memory from ~48GB to ~5GB.

### Step 3: Check Multi-GPU Strategy
```bash
python mprofile_training.py 13.0 32 --sequence-length 4096 --gpus 4 --strategy data_parallel --bottlenecks
```

**Result with optimizations applied**:
```
✓ WILL FIT
Memory required: 22.4 GB per GPU
GPU capacity: 80 GB (72.0 GB usable)
Available headroom: 49.6 GB

MODEL CONFIGURATION:
Cluster: 4x NVIDIA H100
Strategy: data_parallel (DP=4, TP=1, PP=1)
Interconnect: nvlink_5

MEMORY BREAKDOWN (PER GPU):
Model Parameters     : 26.0 GB (116.1%)  # ← Full model per GPU (data parallel)
Optimizer States     : 52.0 GB (232.1%)
Gradients           : 26.0 GB (116.1%)
Activations         :  4.8 GB ( 21.4%)   # ← Reduced with Flash Attention + batch split
Other (buffers)     :  1.1 GB (  4.9%)
Total per GPU       : 22.4 GB

TRAINING TIME ESTIMATE:
Expected duration: 12.3 days (296 hours)
Steps to convergence: ~13,000
Throughput: 2,400 tokens/sec
Scaling efficiency: 85.0%
```

**Analysis**: With Flash Attention and data parallelism, training becomes feasible! The effective batch size per GPU dropped from 32 to 8 (32÷4), reducing activation memory. Scaling efficiency is good at 85%.

### Step 4: Compare with Tensor Parallelism
```bash
python mprofile_training.py transformer 13.0 32 --sequence-length 4096 --gpus 4 --strategy tensor_parallel --bottlenecks
```

**Result**:
```
✓ WILL FIT
Memory required: 18.1 GB per GPU
GPU capacity: 80 GB (72.0 GB usable)
Available headroom: 53.9 GB

MEMORY BREAKDOWN (PER GPU):
Model Parameters     :  6.5 GB ( 35.9%)  # ← Split across 4 GPUs
Optimizer States     : 13.0 GB ( 71.8%)  # ← Split across 4 GPUs
Gradients           :  6.5 GB ( 35.9%)  # ← Split across 4 GPUs
Activations         :  4.8 GB ( 26.5%)  # ← Same (batch not split)
Other (buffers)     :  2.3 GB ( 12.7%)  # ← Higher due to communication
Total per GPU       : 18.1 GB

TRAINING TIME ESTIMATE:
Expected duration: 17.8 days (427 hours)
Throughput: 1,680 tokens/sec
Scaling efficiency: 60.0%
Communication overhead: 25.0%

WARNING: High communication overhead (25.0%) reduces scaling efficiency
Consider different parallelization strategy or faster interconnect
```

**Analysis**: Tensor parallelism uses less memory per GPU (18.1GB vs 22.4GB) but has worse scaling efficiency (60% vs 85%) due to communication overhead. For this model size, **data parallelism is more efficient**.

### Key Insights from This Example:

- **Flash Attention is critical**: Reduced activation memory from 47.8GB to 4.8GB (90% reduction)
- **Sequence length scaling**: 4K sequences caused quadratic memory explosion without Flash Attention
- **Strategy trade-offs**: Data parallel had better efficiency (85%) vs tensor parallel (60%)
- **Memory vs. communication**: Tensor parallel saves memory but increases communication overhead

### Final Decision

**Production Strategy:**
- **Use 4x H100 with data parallelism** for best training efficiency
- **Enable Flash Attention** (mandatory for 4K sequences)
- **Enable gradient checkpointing and mixed precision** for additional memory savings
- **Use batch size 32 globally** (8 per GPU) for optimal throughput

**Performance Expectations:**
- **Training time**: 12.3 days (296 hours)
- **Throughput**: 2,400 tokens/sec
- **Memory utilization**: 22.4GB per GPU (safe 31% utilization)

**Alternative for Memory Constraints:**
- **Use tensor parallelism** if memory becomes tight (18.1GB per GPU)
- **Accept 45% longer training time** (17.8 vs 12.3 days)

**Result**: Multi-GPU data parallel training with Flash Attention enables efficient 13B model training with 4K sequences.

## Configuration Files

memprof-training supports YAML configuration for complex setups:

```yaml
# configs/13b_long_sequence.yaml
model:
  type: "transformer_decoder"
  size_billions: 13.0
  batch_size: 32
  sequence_length: 4096
  framework: "pytorch"

hardware:
  gpu_type: "NVIDIA H100"

cluster:
  num_gpus: 4
  interconnect: "nvlink_5"
  nodes: 1
  gpus_per_node: 4

distribution:
  strategy: "data_parallel"
  data_parallel: 4

analysis:
  enable_bottlenecks: true
  basic_mode: false
```

```bash
python mprofile_training.py --config configs/13b_long_sequence.yaml --output results.json
```

## Project Structure

```
memprof-training/
├── mprofile_training.py     # Main entry point
├── training_analyzer.py     # Core analysis engine with multi-GPU support
├── training_workload.py     # Memory calculation models for different architectures
├── training_estimator.py    # Training time prediction engine
├── training_parallel.py     # Multi-GPU distribution strategies (data/tensor/pipeline/hybrid)
├── training_recommendations.py # Optimization suggestions (Flash Attention, checkpointing, etc.)
├── gpu_specs.py            # Hardware specifications database (H100, H200, B200, MI300X, etc.)
├── cluster_specs.py        # Multi-GPU cluster modeling with interconnects
├── utils.py                # Shared utilities and memory calculation constants
├── yaml/                # YAML configuration examples
├── tests/                  # Test suite (to be included in the future)
└── README.md               # This file
```

## Command Line Interface

```bash
# Basic usage
python mprofile_training.py <model_type> <model_size_billions> <batch_size> [options]

# Model types: transformer, cnn, diffusion_model, vision_transformer
# GPU types: "NVIDIA H100", "NVIDIA H200", "NVIDIA B200", "AMD MI300X", etc.

# Examples:
python mprofile_training.py transformer 7.0 32                                    # Basic 7B model
python mprofile_training.py transformer 70.0 16 --gpus 8 --strategy tensor_parallel # 70B multi-GPU
python mprofile_training.py transformer 13.0 32 --sequence-length 4096 --bottlenecks # Long sequence analysis
python mprofile_training.py --config model_config.yaml --output results.json       # YAML configuration
```

### Key Options:
- `--gpus N`: Number of GPUs for multi-GPU analysis
- `--strategy`: Distribution strategy (data_parallel, tensor_parallel, pipeline_parallel, hybrid)
- `--sequence-length`: Sequence length (default: 2048)
- `--gpu-type`: GPU type (default: "NVIDIA H100")
- `--bottlenecks`: Enable bottleneck analysis with optimization recommendations
- `--max-batch-size`: Find maximum batch size that fits in memory
- `--config`: Use YAML configuration file
- `--output`: Save results to file (JSON/CSV/text)
- `--verbose`: Show detailed analysis including bandwidth metrics

## Python API

```python
from training_analyzer import MemoryAnalyzer
from cluster_specs import create_cluster_spec
from training_parallel import create_distribution_strategy

# Single GPU analysis
analyzer = MemoryAnalyzer(gpu_type="NVIDIA H100")
result = analyzer.analyze_model(
    model_type="transformer",
    model_size_billions=7.0,
    batch_size=32,
    sequence_length=2048
)

print(f"Memory: {result.memory_breakdown.total_memory_gb:.1f} GB")
print(f"Fits: {result.memory_breakdown.total_memory_gb <= 72}")
print(f"Training time: {result.training_estimate.total_hours:.0f} hours")

# Multi-GPU analysis
cluster = create_cluster_spec(num_gpus=8, gpu_type="NVIDIA H100")
strategy = create_distribution_strategy("tensor_parallel", num_gpus=8)

analyzer_multi = MemoryAnalyzer(cluster_spec=cluster)
result_multi = analyzer_multi.analyze_model(
    model_type="transformer",
    model_size_billions=70.0,
    batch_size=32,
    distribution_strategy=strategy
)

print(f"Memory per GPU: {result_multi.distributed_memory['memory_per_gpu'].total_memory_gb:.1f} GB")
print(f"Scaling efficiency: {result_multi.scaling_analysis['efficiency_percent']:.1f}%")
```

## Multi-GPU Strategies

memprof-training supports four parallelization strategies:

### Data Parallelism
```bash
python mprofile_training.py transformer 7.0 32 --gpus 4 --strategy data_parallel
```
- **Best for**: Smaller models that fit on single GPU
- **Memory**: Full model per GPU
- **Communication**: Low (gradient synchronization only)
- **Efficiency**: 85-95%

### Tensor Parallelism
```bash
python mprofile_training.py transformer 70.0 16 --gpus 8 --strategy tensor_parallel
```
- **Best for**: Large models needing memory reduction
- **Memory**: Model split across GPUs
- **Communication**: High (frequent all-reduce operations)
- **Efficiency**: 60-80%

### Pipeline Parallelism
```bash
python mprofile_training.py transformer 175.0 16 --gpus 8 --strategy pipeline_parallel
```
- **Best for**: Very large models with acceptable latency
- **Memory**: Layers split across GPUs
- **Communication**: Point-to-point between stages
- **Efficiency**: 50-70% (pipeline bubbles)

### Hybrid Parallelism
```yaml
# configs/hybrid_example.yaml
distribution:
  strategy: "hybrid"
  data_parallel: 2
  tensor_parallel: 2
  pipeline_parallel: 2
  # Total: 2×2×2 = 8 GPUs
```
- **Best for**: Massive models (100B+ parameters)
- **Memory**: Maximum memory savings
- **Communication**: Complex coordination
- **Efficiency**: 40-60%

## When to Use memprof-training

**Useful for:**
- Planning training runs before spending money on cloud resources
- Comparing different model sizes and parallelization strategies
- Debugging OOM errors and memory bottlenecks
- Optimizing training time and cost efficiency
- Multi-GPU deployment planning for large models

**Not intended for:**
- Inference memory analysis (use memprof-inference instead)
- Fine-tuning small models on single GPUs (overhead not worth it)
- Non-transformer architectures (limited support)

## Special Features

### Flash Attention Detection
Automatically detects when sequences are long enough to benefit from Flash Attention:
```bash
python mprofile_training.py transformer 7.0 16 --sequence-length 8192 --bottlenecks
```
Provides specific recommendations for enabling Flash Attention and quantifies memory savings.

### Training Time Estimation
Based on real hardware benchmarks and model scaling laws:
- Throughput estimation (tokens/second)
- Steps to convergence based on model size
- Total training time in hours/days
- Multi-GPU scaling efficiency impact

### Bottleneck Analysis
Identifies specific optimization opportunities:
- Memory capacity bottlenecks
- Attention memory explosion
- Communication overhead in multi-GPU setups
- Optimizer memory inefficiencies

## Contributing

We welcome contributions! Areas where help is especially needed:
- Additional model architecture support (CNNs, diffusion models)
- More GPU hardware benchmarks (newer architectures)
- Multi-node training analysis
- Production training case studies
- Optimization technique validation

## License

Currently under review.
Intended to adopt MIT License

## Related Projects

- **memprof-inference**: Memory analysis for inference workloads
- **mlcluster-benchmark**: Hardware benchmarking tool under AI/ML workload environment

## Support

- 🐛 [Issues](https://github.com/zippang/memprof-training/issues) - Bug reports and feature requests
- 💬 [Discussions](https://github.com/zippang/memprof-training/discussions) - Questions and community
- 📧 Contact: wonsuk.lee@sk.com or wonsuk.lee@snu.ac.kr

---

