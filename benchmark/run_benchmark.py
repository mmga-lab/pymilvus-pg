#!/usr/bin/env python3
"""
Benchmark runner script that can load configurations from JSON files.
"""

import argparse
import json
import sys
from pathlib import Path

from benchmark_insert_upsert import BenchmarkConfig, InsertUpsertBenchmark


def load_config_from_file(config_file: str) -> BenchmarkConfig:
    """Load benchmark configuration from JSON file."""
    try:
        with open(config_file) as f:
            config_data = json.load(f)

        config = BenchmarkConfig()

        # Update config fields from JSON
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown configuration key '{key}' ignored")

        return config

    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def list_available_configs():
    """List all available configuration files."""
    configs_dir = Path(__file__).parent / "configs"
    if not configs_dir.exists():
        print("No configs directory found")
        return

    config_files = list(configs_dir.glob("*.json"))
    if not config_files:
        print("No configuration files found in configs directory")
        return

    print("Available configuration presets:")
    for config_file in sorted(config_files):
        try:
            with open(config_file) as f:
                config_data = json.load(f)

            description = []
            if "record_counts" in config_data:
                counts = config_data["record_counts"]
                description.append(f"{min(counts)}-{max(counts)} records")
            if "vector_dims" in config_data:
                dims = config_data["vector_dims"]
                description.append(f"{min(dims)}-{max(dims)}D vectors")
            if "concurrent_workers" in config_data:
                workers = config_data["concurrent_workers"]
                description.append(f"{min(workers)}-{max(workers)} workers")

            desc_str = ", ".join(description) if description else "Custom configuration"
            print(f"  {config_file.stem}: {desc_str}")

        except Exception as e:
            print(f"  {config_file.stem}: Error reading config - {e}")


def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Run PyMilvus-PG benchmarks with predefined configurations")

    parser.add_argument("--config", type=str, help="Configuration file to use (JSON format)")
    parser.add_argument("--list-configs", action="store_true", help="List available configuration presets")
    parser.add_argument("--output", type=str, help="Output file for results (defaults to config-specific name)")

    args = parser.parse_args()

    if args.list_configs:
        list_available_configs()
        return

    if not args.config:
        print("Error: Please specify a configuration file with --config or use --list-configs to see available options")
        sys.exit(1)

    # Load configuration
    config = load_config_from_file(args.config)

    # Determine output file name
    if args.output:
        output_file = args.output
    else:
        config_name = Path(args.config).stem
        output_file = f"benchmark_results_{config_name}.json"

    print(f"Running benchmark with configuration: {args.config}")
    print(f"Results will be saved to: {output_file}")
    print("-" * 60)

    # Run benchmark
    benchmark = InsertUpsertBenchmark(config)
    benchmark.run_benchmark_suite()

    # Export results and generate report
    benchmark.export_results(output_file)
    benchmark.generate_summary_report()


if __name__ == "__main__":
    main()
