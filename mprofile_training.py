#!/usr/bin/env python

import os
import sys
import argparse
import json
import yaml
import logging
from typing import Dict, Any, Optional, List

from training_analyzer import MemoryAnalyzer, AnalysisResult
from utils import AnalysisType
from training_recommendations import RecommendationEngine
from cluster_specs import ClusterSpec, create_cluster_spec
from training_parallel import DistributionStrategy, create_distribution_strategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MemoryProfiler:
    """Enhanced command-line interface for memory analysis with training time estimation."""

    def __init__(self):
        """Initialize memory profiler."""
        self.analyzer = None
        self.recommendation_engine = None

    def run_analysis(self, args) -> None:
        """Run memory analysis based on command line arguments."""
        try:
            # Load configuration from YAML if provided
            if args.config:
                config = self._load_yaml_config(args.config)
                # Override config with command line arguments
                self._merge_args_with_config(args, config)

            # Extract multi-GPU configuration
            cluster_spec = None
            distribution_strategy = None

            # Check for multi-GPU setup
            if hasattr(args, 'num_gpus') and args.num_gpus > 1:
                # Command line multi-GPU
                cluster_spec = create_cluster_spec(args.num_gpus, args.gpu_type)
                if hasattr(args, 'strategy') and args.strategy:
                    distribution_strategy = create_distribution_strategy(args.strategy, args.num_gpus)
            elif hasattr(args, 'cluster_config') and args.cluster_config:
                # YAML multi-GPU configuration
                cluster_config = args.cluster_config
                distribution_config = getattr(args, 'distribution_config', {})

                cluster_spec = ClusterSpec(
                    num_gpus=cluster_config.get('num_gpus', 1),
                    gpu_type=args.gpu_type,  # Use GPU type from hardware section
                    interconnect=cluster_config.get('interconnect', 'nvlink_5'),
                    nodes=cluster_config.get('nodes', 1),
                    gpus_per_node=cluster_config.get('gpus_per_node')
                )

                if distribution_config.get('strategy'):
                    strategy_name = distribution_config['strategy']

                    if strategy_name == "hybrid":
                        # FIXED: Handle hybrid strategy with explicit parameters from YAML
                        distribution_strategy = DistributionStrategy(
                            data_parallel=distribution_config.get('data_parallel', 1),
                            tensor_parallel=distribution_config.get('tensor_parallel', 1),
                            pipeline_parallel=distribution_config.get('pipeline_parallel', 1),
                            model_parallel=distribution_config.get('model_parallel', 1)
                        )
                    else:
                        # Use the factory function for single strategies
                        distribution_strategy = create_distribution_strategy(
                            strategy_name,
                            cluster_spec.num_gpus,
                            data_parallel=distribution_config.get('data_parallel'),
                            tensor_parallel=distribution_config.get('tensor_parallel'),
                            pipeline_parallel=distribution_config.get('pipeline_parallel'),
                            model_parallel=distribution_config.get('model_parallel')
                        )

            # Initialize analyzer with cluster support
            self.analyzer = MemoryAnalyzer(
                framework=args.framework,
                gpu_type=args.gpu_type,
                enable_detailed_analysis=not args.basic,
                cluster_spec=cluster_spec
            )

            # Initialize recommendation engine with fixed infrastructure focus
            self.recommendation_engine = RecommendationEngine(
                framework=args.framework,
                fixed_infrastructure=True
            )

            # Determine analysis type
            analysis_type = AnalysisType.BOTTLENECK if args.bottlenecks else AnalysisType.PROGRESSIVE

            # Log analysis details
            if cluster_spec and cluster_spec.num_gpus > 1:
                logger.info(f"Running {analysis_type.value} analysis with {cluster_spec.num_gpus}x {cluster_spec.gpu_type}")
                if distribution_strategy:
                    logger.info(f"Distribution strategy: {distribution_strategy.primary_strategy.value}")
                    if distribution_strategy.primary_strategy.value == "hybrid":
                        logger.info(f"Hybrid configuration: DP={distribution_strategy.data_parallel}, "
                                  f"TP={distribution_strategy.tensor_parallel}, "
                                  f"PP={distribution_strategy.pipeline_parallel}")
            else:
                logger.info(f"Running {analysis_type.value} analysis...")

            logger.info(f"Model: {args.model_type} ({args.model_size}B params)")
            logger.info(f"Config: batch_size={args.batch_size}, seq_length={args.sequence_length}")

            # Run analysis
            result = self.analyzer.analyze_model(
                model_type=args.model_type,
                model_size_billions=args.model_size,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                hidden_dim=args.hidden_dim,
                analysis_type=analysis_type,
                distribution_strategy=distribution_strategy
            )

            # Print results
            if not args.quiet:
                self._print_enhanced_results(result, args)

            # Save results
            if args.output:
                self._save_results(result, args.output, args.format)
                logger.info(f"Results saved to {args.output}")

            # Show max batch size if requested
            if args.max_batch_size:
                max_batch = self.analyzer.get_max_batch_size(
                    args.model_type, args.model_size, args.sequence_length,
                    distribution_strategy=distribution_strategy
                )
                print(f"\nMaximum batch size: {max_batch}")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    def _print_enhanced_results(self, result: AnalysisResult, args) -> None:
        """Enhanced output with feasibility assessment and training time estimation."""

        print("\n" + "="*80)
        print("TRAINING ANALYSIS RESULTS")
        print("="*80)

        # 1. FEASIBILITY ASSESSMENT - Clear go/no-go decision
        feasibility_status = self._print_feasibility_assessment(result)

        # 2. MODEL AND HARDWARE INFO
        self._print_model_info(result)

        # 3. MEMORY BREAKDOWN - Condensed version
        self._print_memory_breakdown(result)

        # 4. ACTIONABLE RECOMMENDATIONS (for infeasible configs)
        if result.recommendations:
            self._print_actionable_recommendations(result.recommendations, args)

        # 5. TRAINING TIME ESTIMATE - Only for feasible configs or after recommendations
        if result.training_estimate:
            self._print_conditional_training_estimate(result.training_estimate, feasibility_status)

        # 6. DETAILED METRICS (if verbose)
        if args.verbose and result.bandwidth_metrics:
            self._print_detailed_metrics(result)

        print("="*80)

    def _print_feasibility_assessment(self, result: AnalysisResult) -> bool:
        """Print clear feasibility assessment and return feasibility status."""
        # Get memory usage (distributed or single GPU)
        if result.distributed_memory:
            memory_gb = result.distributed_memory["memory_per_gpu"].total_memory_gb
        else:
            memory_gb = result.memory_breakdown.total_memory_gb

        gpu_capacity = self.analyzer.gpu_spec.memory_capacity_gb
        fits = memory_gb <= gpu_capacity * 0.9

        # Determine status and message
        if fits:
            status = "✓ WILL FIT"
            status_color = ""
            headroom = gpu_capacity * 0.9 - memory_gb
            detail = f"Available headroom: {headroom:.1f} GB"
        else:
            status = "✗ EXCEEDS CAPACITY"
            status_color = ""
            overflow = memory_gb - gpu_capacity * 0.9
            detail = f"Memory overflow: {overflow:.1f} GB (optimization required)"

        print(f"\nFEASIBILITY: {status}")
        print(f"Memory required: {memory_gb:.1f} GB per GPU")
        print(f"GPU capacity: {gpu_capacity} GB ({gpu_capacity * 0.9:.1f} GB usable)")
        print(f"{detail}")

        # Add cluster context if multi-GPU
        if result.cluster_spec and result.cluster_spec.num_gpus > 1:
            total_memory = memory_gb * result.cluster_spec.num_gpus
            print(f"Total cluster memory: {total_memory:.1f} GB across {result.cluster_spec.num_gpus} GPUs")

        return fits  # Return feasibility status

    def _print_conditional_training_estimate(self, estimate, is_feasible: bool) -> None:
        """Print training time estimation with appropriate context."""
        days = estimate.total_hours / 24

        if is_feasible:
            # Configuration works as-is
            print(f"\nTRAINING TIME ESTIMATE:")
            print(f"Expected duration: {days:.1f} days ({estimate.total_hours:.0f} hours)")
            print(f"Steps to convergence: ~{estimate.steps_to_convergence:,}")
            print(f"Throughput: {estimate.throughput_tokens_per_sec:,} tokens/sec")
        else:
            # Configuration needs optimization
            print(f"\nESTIMATED TRAINING TIME (After Applying Optimizations):")
            print(f"Once memory issues are resolved with above recommendations:")
            print(f"Expected duration: {days:.1f} days ({estimate.total_hours:.0f} hours)")
            print(f"Steps to convergence: ~{estimate.steps_to_convergence:,}")
            print(f"Throughput: {estimate.throughput_tokens_per_sec:,} tokens/sec")
            print(f"")
            print(f"Note: This estimate assumes gradient checkpointing is enabled")
            print(f"      (which may increase training time by 20-30%)")

    def _print_training_estimate(self, estimate) -> None:
        """Print training time estimation (legacy method for compatibility)."""
        days = estimate.total_hours / 24

        print(f"\nTRAINING TIME ESTIMATE:")
        print(f"Expected duration: {days:.1f} days ({estimate.total_hours:.0f} hours)")
        print(f"Steps to convergence: ~{estimate.steps_to_convergence:,}")
        print(f"Throughput: {estimate.throughput_tokens_per_sec:,} tokens/sec")

    def _print_model_info(self, result: AnalysisResult) -> None:
        """Print model and hardware configuration."""
        model_info = result.model_info

        print(f"\nMODEL CONFIGURATION:")
        print(f"Type: {model_info['model_type']} ({model_info['model_size_billions']}B parameters)")
        print(f"Batch size: {model_info['batch_size']}")
        print(f"Sequence length: {model_info['sequence_length']}")

        if result.cluster_spec and result.cluster_spec.num_gpus > 1:
            print(f"Cluster: {result.cluster_spec.num_gpus}x {result.cluster_spec.gpu_type}")
            if result.distribution_strategy:
                strategy_info = f"{result.distribution_strategy.primary_strategy.value}"
                if result.distribution_strategy.primary_strategy.value == "hybrid":
                    strategy_info += f" (DP={result.distribution_strategy.data_parallel}, "
                    strategy_info += f"TP={result.distribution_strategy.tensor_parallel}, "
                    strategy_info += f"PP={result.distribution_strategy.pipeline_parallel})"
                print(f"Strategy: {strategy_info}")
            print(f"Interconnect: {result.cluster_spec.interconnect}")
        else:
            print(f"GPU: {model_info['gpu_type']}")

    def _print_memory_breakdown(self, result: AnalysisResult) -> None:
        """Print condensed memory breakdown."""
        print(f"\nMEMORY BREAKDOWN:")
        print("-" * 50)

        # Use distributed memory if available
        if result.distributed_memory:
            memory = result.distributed_memory["memory_per_gpu"]
            print("PER-GPU MEMORY USAGE:")
        else:
            memory = result.memory_breakdown
            print("MEMORY USAGE:")

        # Show only the most important components
        key_components = [
            ("Model Parameters", memory.model_parameters_gb),
            ("Optimizer States", memory.optimizer_states_gb),
            ("Gradients", memory.gradients_gb),
            ("Activations", memory.activations_gb),
            ("Other (buffers + overhead)", memory.temporary_buffers_gb + memory.framework_overhead_gb + memory.fragmentation_gb)
        ]

        for name, size_gb in key_components:
            percentage = (size_gb / memory.total_memory_gb * 100) if memory.total_memory_gb > 0 else 0
            print(f"{name:20s}: {size_gb:8.2f} GB ({percentage:5.1f}%)")

        print("-" * 50)
        print(f"{'Total per GPU':20s}: {memory.total_memory_gb:8.2f} GB")

        # GPU utilization
        gpu_capacity = self.analyzer.gpu_spec.memory_capacity_gb
        utilization = (memory.total_memory_gb / gpu_capacity * 100) if gpu_capacity > 0 else 0
        print(f"GPU utilization: {utilization:.1f}% of {gpu_capacity:.0f} GB capacity")

        # ADD THIS: Multi-GPU scaling efficiency (always show for multi-GPU)
        if result.cluster_spec and result.cluster_spec.num_gpus > 1:
            if result.scaling_analysis:
                scaling_eff = result.scaling_analysis.get("efficiency_percent", 70.0)
            elif result.distributed_memory:
                parallelization_eff = result.distributed_memory.get("parallelization_efficiency", 0.7)
                scaling_eff = parallelization_eff * 100
            else:
                scaling_eff = 70.0  # Default
            print(f"Scaling efficiency: {scaling_eff:.1f}%")

    def _print_actionable_recommendations(self, recommendations: List[Dict], args) -> None:
        """Print actionable recommendations focused on software optimizations with strategic direction."""
        # Filter recommendations by priority for better display
        critical_recs = [r for r in recommendations if r.get("priority") == "critical"]
        high_recs = [r for r in recommendations if r.get("priority") == "high"]
        info_recs = [r for r in recommendations if r.get("priority") == "info"]

        if not (critical_recs or high_recs):
            return

        print(f"\nRECOMMENDATIONS:")
        print("-" * 50)

        # Show critical first, then high, then important info recs
        important_info_recs = [r for r in info_recs if "Combined Optimization" in r.get("title", "")]
        other_info_recs = [r for r in info_recs if "Combined Optimization" not in r.get("title", "")]

        all_recs = critical_recs + high_recs + important_info_recs + other_info_recs[:1]

        for i, rec in enumerate(all_recs, 1):
            priority = rec.get('priority', 'medium').upper()
            title = rec['title']

            print(f"{i}. [{priority}] {title}")
            if rec.get('description'):
                print(f"   {rec.get('description', '')}")

            # Show key techniques (filter out empty ones)
            techniques = [t for t in rec.get('techniques', []) if t.strip()]
            for technique in techniques:
                print(f"   • {technique}")

            # Show expected savings
            if 'estimated_savings' in rec:
                print(f"   Expected: {rec['estimated_savings']}")

            print()

        # Show quick wins if available and verbose
        if args.verbose:
            quick_wins = self.recommendation_engine.get_quick_wins(recommendations)
            if quick_wins and len(quick_wins) != len(all_recs):
                print("QUICK WINS (High impact, low effort):")
                for rec in quick_wins[:2]:
                    print(f"  • {rec['title']}")
                print()

    def _print_detailed_metrics(self, result: AnalysisResult) -> None:
        """Print detailed performance metrics (verbose mode)."""
        print(f"DETAILED METRICS:")
        print("-" * 50)

        bw = result.bandwidth_metrics
        print(f"Memory bandwidth: {bw.get('theoretical_bandwidth_gb_s', 0):.0f} GB/s theoretical")
        print(f"Effective bandwidth: {bw.get('effective_bandwidth_gb_s', 0):.0f} GB/s")

        if 'bandwidth_utilization' in bw:
            print(f"Bandwidth utilization: {bw['bandwidth_utilization']:.1%}")

        # Multi-GPU metrics
        if result.cluster_spec and result.cluster_spec.num_gpus > 1:
            print(f"Cluster bandwidth: {bw.get('cluster_total_bandwidth_gb_s', 0):.0f} GB/s")
            print(f"Interconnect: {bw.get('interconnect_bandwidth_gb_s', 0):.0f} GB/s")

            if result.communication_analysis:
                comm_overhead = result.communication_analysis.get("communication_overhead_percent", 0)
                print(f"Communication overhead: {comm_overhead:.1f}%")

            if result.scaling_analysis:
                scaling_eff = result.scaling_analysis.get("efficiency_percent", 0)
                print(f"Scaling efficiency: {scaling_eff:.1f}%")

    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Validate required sections
            required_sections = ["model"]
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required section '{section}' in config file")
                    sys.exit(1)

            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            sys.exit(1)

    def _merge_args_with_config(self, args, config: Dict[str, Any]) -> None:
        """Merge command line arguments with YAML config, including multi-GPU sections."""
        model_config = config.get("model", {})
        hardware_config = config.get("hardware", {})
        analysis_config = config.get("analysis", {})

        # Multi-GPU configurations
        cluster_config = config.get("cluster", {})
        distribution_config = config.get("distribution", {})

        # Override args with config values if not provided via command line
        if not hasattr(args, 'model_type') or args.model_type is None:
            args.model_type = model_config.get("type", "transformer")
        if not hasattr(args, 'model_size') or args.model_size is None:
            args.model_size = model_config.get("size_billions", 1.0)
        if not hasattr(args, 'batch_size') or args.batch_size is None:
            args.batch_size = model_config.get("batch_size", 32)

        # Optional parameters from config (YAML values take precedence over CLI defaults)
        args.sequence_length = model_config.get("sequence_length", getattr(args, 'sequence_length', 2048))
        args.hidden_dim = model_config.get("hidden_dim", getattr(args, 'hidden_dim', None))
        args.framework = model_config.get("framework", getattr(args, 'framework', "pytorch"))
        args.gpu_type = hardware_config.get("gpu_type", getattr(args, 'gpu_type', "NVIDIA H100"))

        # Analysis options
        if not getattr(args, 'bottlenecks', False):
            args.bottlenecks = analysis_config.get("enable_bottlenecks", False)
        if not getattr(args, 'basic', False):
            args.basic = analysis_config.get("basic_mode", False)

        # Multi-GPU configurations - FIXED to properly handle hybrid
        if cluster_config:
            args.cluster_config = cluster_config
            # GPU type comes from hardware section, not cluster section

        if distribution_config:
            args.distribution_config = distribution_config

    def _save_results(self, result: AnalysisResult, output_path: str, format: str) -> None:
        """Save results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Component", "Memory_GB", "Percentage"])

                # Use distributed memory if available
                if result.distributed_memory:
                    memory = result.distributed_memory["memory_per_gpu"]
                else:
                    memory = result.memory_breakdown

                total = memory.total_memory_gb

                components = [
                    ("model_parameters", memory.model_parameters_gb),
                    ("optimizer_states", memory.optimizer_states_gb),
                    ("gradients", memory.gradients_gb),
                    ("activations", memory.activations_gb),
                    ("temporary_buffers", memory.temporary_buffers_gb),
                    ("framework_overhead", memory.framework_overhead_gb),
                    ("fragmentation", memory.fragmentation_gb),
                    ("total", total)
                ]

                for name, size in components:
                    pct = (size / total * 100) if total > 0 else 0
                    writer.writerow([name, f"{size:.2f}", f"{pct:.1f}"])

        else:  # text format
            with open(output_path, 'w') as f:
                f.write(result.get_summary())
                f.write("\n\nDetailed Results:\n")
                f.write(json.dumps(result.to_dict(), indent=2))

def create_parser():
    """Create command line argument parser with enhanced options."""
    parser = argparse.ArgumentParser(
        description="Memory Performance Analysis Tool (mprof) with Training Time Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with training time estimation
  python mproffile-training.py transformer 7.0 32

  # Multi-GPU analysis
  python mproffile-training.py transformer 70.0 16 --gpus 8 --strategy tensor_parallel

  # Using YAML config file (supports hybrid strategies)
  python mproffile-training.py --config model_config.yaml

  # Full analysis with recommendations
  python mproffile-training.py transformer 7.0 32 --bottlenecks --verbose

  # Save results
  python mproffile-training.py transformer 7.0 32 --output results.json --format json
        """
    )

    # Configuration file
    parser.add_argument("--config", help="YAML configuration file path")

    # Required arguments (optional if using config file)
    parser.add_argument("model_type", nargs='?', help="Model type (transformer, cnn, etc.)")
    parser.add_argument("model_size", type=float, nargs='?', help="Model size in billions of parameters")
    parser.add_argument("batch_size", type=int, nargs='?', help="Training batch size")

    # Optional model parameters
    parser.add_argument("--sequence-length", type=int, default=2048,
                       help="Sequence length (default: 2048)")
    parser.add_argument("--hidden-dim", type=int,
                       help="Hidden dimension (auto-estimated if not provided)")

    # Hardware and framework
    parser.add_argument("--gpu-type", default="NVIDIA H100",
                       help="GPU type (default: NVIDIA H100)")
    parser.add_argument("--framework", default="pytorch",
                       choices=["pytorch", "tensorflow", "jax"],
                       help="ML framework (default: pytorch)")

    # Multi-GPU options
    parser.add_argument("--gpus", type=int, dest="num_gpus", default=1,
                       help="Number of GPUs for multi-GPU analysis (default: 1)")
    parser.add_argument("--strategy", choices=["data_parallel", "tensor_parallel", "pipeline_parallel", "hybrid"],
                       help="Multi-GPU distribution strategy")

    # Analysis options
    parser.add_argument("--bottlenecks", action="store_true",
                       help="Run bottleneck analysis with recommendations")
    parser.add_argument("--basic", action="store_true",
                       help="Run basic analysis only (faster)")
    parser.add_argument("--max-batch-size", action="store_true",
                       help="Find maximum batch size that fits in GPU memory")

    # Output options
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["json", "csv", "text"], default="json",
                       help="Output format (default: json)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress console output")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed output including metrics")

    return parser

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if not args.config and (not args.model_type or args.model_size is None or args.batch_size is None):
        parser.error("Either --config must be provided, or model_type, model_size, and batch_size must be specified")

    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run analysis
    profiler = MemoryProfiler()
    profiler.run_analysis(args)

if __name__ == "__main__":
    main()