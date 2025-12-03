#!/usr/bin/env python3
"""
Visualization script for time-axis partitioning fusion decoder.

This script generates comprehensive visualizations from benchmark output, including:
- Partition quality metrics (boundary ratios, comparison between strategies)
- Decoding performance (success rates, iterations, timing)
- Code structure (circulant blocks, Tanner graph, partition boundaries)
- Boundary variable analysis
- Sparse matrix statistics

The script can either:
1. Run the benchmark automatically and visualize results
2. Use saved benchmark output (with --no-run flag)

Requirements:
    Install dependencies: uv pip install -r requirements.txt
    Or: pip install -r requirements.txt

Usage:
    python visualize_fusion.py                    # Run benchmark and visualize
    python visualize_fusion.py --no-run          # Use saved benchmark output
    python visualize_fusion.py --output-dir ./out # Custom output directory
"""

import subprocess
import re
import json
import sys
import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.sparse import load_npz
import networkx as nx

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10

# Color scheme constants
COLOR_TIME_AXIS = '#2ecc71'  # Green
COLOR_COUNT_BASED = '#e74c3c'  # Red
COLOR_BASELINE = '#3498db'  # Blue
COLOR_PALETTE = [COLOR_BASELINE, COLOR_TIME_AXIS, COLOR_COUNT_BASED]


def parse_benchmark_output(output: str) -> Dict:
    """
    Parse benchmark output text to extract structured data.
    
    Extracts:
    - Code information (dimensions, rounds, etc.)
    - Partition quality metrics
    - Decoding performance metrics
    - Agreement statistics
    
    Args:
        output: Raw benchmark output text
        
    Returns:
        Dictionary containing parsed metrics and data
    """
    data = {
        'partition_analyses': [],
        'baseline_metrics': {},
        'time_axis_metrics': {},
        'count_based_metrics': {},
        'validation_results': {},
        'code_info': {},
    }
    
    # Parse code information
    for line in output.split('\n'):
        if 'Code:' in line:
            # Extract code type, e.g., "Code: [[72, 12, 6]] bivariate bicycle"
            match = re.search(r'\[\[(\d+),\s*(\d+),\s*(\d+)\]\]', line)
            if match:
                data['code_info']['distance'] = int(match.group(3))
                data['code_info']['logical_qubits'] = int(match.group(2))
                data['code_info']['physical_qubits'] = int(match.group(1))
        if 'Check matrix:' in line:
            # Extract dimensions, e.g., "Check matrix: 432 detectors × 16164 error variables"
            match = re.search(r'(\d+)\s+detectors\s+×\s+(\d+)\s+error variables', line)
            if match:
                data['code_info']['num_detectors'] = int(match.group(1))
                data['code_info']['num_variables'] = int(match.group(2))
        if 'Detectors per round:' in line:
            match = re.search(r'Detectors per round:\s+(\d+)', line)
            if match:
                data['code_info']['detectors_per_round'] = int(match.group(1))
        if 'Total rounds:' in line:
            match = re.search(r'Total rounds:\s+(\d+)', line)
            if match:
                data['code_info']['total_rounds'] = int(match.group(1))
    
    # Parse partition quality table
    in_table = False
    for line in output.split('\n'):
        if 'Partition Quality Results:' in line:
            in_table = True
            continue
        # Stop parsing if we hit a new section (starts with [number/number])
        if in_table and re.match(r'^\[.*\]', line.strip()):
            in_table = False
            continue
        if in_table and '|' in line and 'Strategy' not in line and '---' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                try:
                    data['partition_analyses'].append({
                        'strategy': parts[0],
                        'num_partitions': int(parts[1]),
                        'avg_boundary_ratio': float(parts[2].replace('%', '')) / 100.0,
                        'max_boundary_ratio': float(parts[3].replace('%', '')) / 100.0,
                        'min_boundary_ratio': float(parts[4].replace('%', '')) / 100.0,
                    })
                except (ValueError, IndexError):
                    # Skip lines that don't match the expected format
                    continue
    
    # Parse correctness results
    for line in output.split('\n'):
        if 'Baseline' in line and '|' in line and 'Decoder' not in line and 'Success Rate' not in line:
            parts = [p.strip() for p in line.split('|')]
            # Parse metrics (supports format with or without timing column)
            if len(parts) >= 4:
                try:
                    success_parts = parts[1].split('/')
                    syndrome_parts = parts[2].split('/')
                    if len(success_parts) == 2 and len(syndrome_parts) == 2:
                        data['baseline_metrics'] = {
                            'success_count': int(success_parts[0]),
                            'total_samples': int(success_parts[1]),
                            'syndrome_satisfaction': int(syndrome_parts[0]),
                            'avg_iterations': float(parts[3]),
                            'avg_time_ms': float(parts[4]) if len(parts) > 4 and parts[4] else 0.0,
                        }
                except (ValueError, IndexError):
                    continue
        elif 'Time-axis fusion' in line and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                try:
                    success_parts = parts[1].split('/')
                    syndrome_parts = parts[2].split('/')
                    if len(success_parts) == 2 and len(syndrome_parts) == 2:
                        data['time_axis_metrics'] = {
                            'success_count': int(success_parts[0]),
                            'total_samples': int(success_parts[1]),
                            'syndrome_satisfaction': int(syndrome_parts[0]),
                            'avg_iterations': float(parts[3]),
                            'avg_time_ms': float(parts[4]) if len(parts) > 4 and parts[4] else 0.0,
                        }
                except (ValueError, IndexError):
                    continue
        elif 'Count-based fusion' in line and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                try:
                    success_parts = parts[1].split('/')
                    syndrome_parts = parts[2].split('/')
                    if len(success_parts) == 2 and len(syndrome_parts) == 2:
                        data['count_based_metrics'] = {
                            'success_count': int(success_parts[0]),
                            'total_samples': int(success_parts[1]),
                            'syndrome_satisfaction': int(syndrome_parts[0]),
                            'avg_iterations': float(parts[3]),
                            'avg_time_ms': float(parts[4]) if len(parts) > 4 and parts[4] else 0.0,
                        }
                except (ValueError, IndexError):
                    continue
    
    # Parse agreement data
    for line in output.split('\n'):
        if 'Time-axis vs Baseline:' in line:
            match = re.search(r'(\d+)/(\d+)', line)
            if match:
                data['agreement'] = data.get('agreement', {})
                data['agreement']['time_axis_vs_baseline'] = {
                    'identical': int(match.group(1)),
                    'total': int(match.group(2))
                }
        elif 'Count-based vs Baseline:' in line:
            match = re.search(r'(\d+)/(\d+)', line)
            if match:
                data['agreement'] = data.get('agreement', {})
                data['agreement']['count_based_vs_baseline'] = {
                    'identical': int(match.group(1)),
                    'total': int(match.group(2))
                }
    
    return data


def run_benchmark() -> str:
    """
    Run the benchmark example and capture its output.
    
    Returns:
        Benchmark output as string
        
    Raises:
        FileNotFoundError: If benchmark file or Cargo.toml not found
        RuntimeError: If benchmark execution fails
    """
    script_dir = Path(__file__).parent
    benchmark_path = script_dir / "benchmark_time_axis_partitioning.rs"
    
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
    
    # Find project root (where Cargo.toml is)
    project_root = script_dir.parent.parent.parent
    cargo_toml = project_root / "Cargo.toml"
    
    if not cargo_toml.exists():
        # Check alternative location
        project_root = script_dir.parent.parent
        cargo_toml = project_root / "Cargo.toml"
        if not cargo_toml.exists():
            raise FileNotFoundError(f"Could not find Cargo.toml. Tried: {script_dir.parent.parent.parent} and {script_dir.parent.parent}")
    
    # Run the benchmark using cargo run --example
    cmd = ["cargo", "run", "--example", "benchmark_time_axis_partitioning", "--release"]
    
    print("Running benchmark...")
    print(f"Working directory: {project_root}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Error running benchmark:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("Benchmark failed")
    
    return result.stdout


def visualize_partition_quality(data: Dict, output_dir: Path) -> None:
    """
    Create visualization comparing partition quality metrics.
    
    Generates plots showing boundary ratios for different partitioning strategies,
    comparing time-axis vs count-based approaches.
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save visualization
    """
    if not data['partition_analyses']:
        print("No partition analysis data found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Partition Quality Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    analyses = data['partition_analyses']
    time_axis = [a for a in analyses if a['strategy'].startswith('Time-axis')]
    count_based = [a for a in analyses if a['strategy'].startswith('Count-based')]
    
    # Plot 1: Average boundary ratio comparison by number of partitions
    # This is the fair comparison - both strategies with the same number of partitions
    ax = axes[0, 0]
    
    # Sort both by number of partitions for fair comparison
    time_axis_sorted = sorted(time_axis, key=lambda x: x['num_partitions'])
    count_based_sorted = sorted(count_based, key=lambda x: x['num_partitions'])
    
    if time_axis_sorted:
        time_partitions = [a['num_partitions'] for a in time_axis_sorted]
        time_ratios = [a['avg_boundary_ratio'] * 100 for a in time_axis_sorted]
        ax.plot(time_partitions, time_ratios, 'o-', label='Time-axis', linewidth=2.5, 
               markersize=10, color=COLOR_TIME_AXIS, markerfacecolor=COLOR_TIME_AXIS,
               markeredgecolor='white', markeredgewidth=1.5)
    
    if count_based_sorted:
        count_partitions = [a['num_partitions'] for a in count_based_sorted]
        count_ratios = [a['avg_boundary_ratio'] * 100 for a in count_based_sorted]
        ax.plot(count_partitions, count_ratios, 's-', label='Count-based', linewidth=2.5,
               markersize=10, color=COLOR_COUNT_BASED, markerfacecolor=COLOR_COUNT_BASED,
               markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('Number of Partitions', fontweight='bold', fontsize=11)
    ax.set_ylabel('Average Boundary Ratio (%)', fontweight='bold', fontsize=11)
    ax.set_title('Boundary Ratio vs Number of Partitions\n(Fair comparison: same partition count)', 
                 fontweight='bold', fontsize=12, pad=10)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Plot 2: Min/Max boundary ratio range
    ax = axes[0, 1]
    strategies = [a['strategy'] for a in analyses]
    x_pos = np.arange(len(strategies))
    avg_ratios = [a['avg_boundary_ratio'] * 100 for a in analyses]
    min_ratios = [a['min_boundary_ratio'] * 100 for a in analyses]
    max_ratios = [a['max_boundary_ratio'] * 100 for a in analyses]
    
    colors = [COLOR_TIME_AXIS if 'Time-axis' in s else COLOR_COUNT_BASED for s in strategies]
    bars = ax.bar(x_pos, avg_ratios, yerr=[np.array(avg_ratios) - np.array(min_ratios),
                                            np.array(max_ratios) - np.array(avg_ratios)],
                  capsize=6, alpha=0.8, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.split('(')[0].strip() for s in strategies], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Boundary Ratio (%)', fontweight='bold', fontsize=11)
    ax.set_title('Boundary Ratio Range (Min/Avg/Max)', fontweight='bold', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Plot 3: Time-axis rounds per partition vs boundary ratio
    # This shows how time-axis partitioning quality changes with rounds per partition
    ax = axes[1, 0]
    
    if time_axis:
        # Extract rounds per partition from strategy name
        time_axis_with_rounds = []
        for a in time_axis:
            match = re.search(r'(\d+)', a['strategy'])
            if match:
                rounds_per_part = int(match.group(1))
                time_axis_with_rounds.append((rounds_per_part, a))
        time_axis_with_rounds.sort(key=lambda x: x[0])
        
        rounds_per_part = [r for r, _ in time_axis_with_rounds]
        time_ratios = [a['avg_boundary_ratio'] * 100 for _, a in time_axis_with_rounds]
        
        ax.plot(rounds_per_part, time_ratios, 'o-', label='Time-axis', linewidth=2.5,
               markersize=10, color=COLOR_TIME_AXIS, markerfacecolor=COLOR_TIME_AXIS,
               markeredgecolor='white', markeredgewidth=1.5)
        
        # Add annotations showing resulting partition count
        for (r, a), ratio in zip(time_axis_with_rounds, time_ratios):
            ax.annotate(f"{a['num_partitions']} parts", 
                       xy=(r, ratio), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Rounds per Partition (Time-axis only)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Average Boundary Ratio (%)', fontweight='bold', fontsize=11)
    ax.set_title('Time-axis: Rounds/Partition vs Boundary Ratio', 
                 fontweight='bold', fontsize=12, pad=10)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Plot 4: Improvement factor
    ax = axes[1, 1]
    if time_axis and count_based:
        best_time = min(time_axis, key=lambda x: x['avg_boundary_ratio'])
        best_count = min(count_based, key=lambda x: x['avg_boundary_ratio'])
        
        # Handle division by zero (when best_time ratio is 0, it means perfect partitioning)
        if best_time['avg_boundary_ratio'] > 0:
            improvement = best_count['avg_boundary_ratio'] / best_time['avg_boundary_ratio']
            improvement_text = f'Improvement: {improvement:.2f}x'
        else:
            improvement = float('inf') if best_count['avg_boundary_ratio'] > 0 else 1.0
            improvement_text = 'Improvement: ∞ (perfect time-axis)' if best_count['avg_boundary_ratio'] > 0 else 'Both perfect'
        
        categories = ['Best Time-axis', 'Best Count-based']
        ratios = [best_time['avg_boundary_ratio'] * 100, best_count['avg_boundary_ratio'] * 100]
        colors_bar = [COLOR_TIME_AXIS, COLOR_COUNT_BASED]
        
        bars = ax.bar(categories, ratios, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Boundary Ratio (%)', fontweight='bold', fontsize=11)
        ax.set_title(f'Best Strategy Comparison\n({improvement_text})', fontweight='bold', fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(ratios) * 0.02,
                   f'{ratio:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor comparison', 
               ha='center', va='center', fontsize=12,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_dir / 'partition_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'partition_quality.png'}")


def visualize_decoding_performance(data: Dict, output_dir: Path) -> None:
    """
    Create visualization of decoding performance metrics.
    
    Compares success rates, iterations, and timing across baseline,
    time-axis fusion, and count-based fusion decoders.
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Decoding Performance', fontsize=16, fontweight='bold', y=0.98)
    
    decoders = ['Baseline', 'Time-axis\nFusion', 'Count-based\nFusion']
    metrics_data = [
        data.get('baseline_metrics', {}),
        data.get('time_axis_metrics', {}),
        data.get('count_based_metrics', {}),
    ]
    
    # Extract all metrics
    success_rates = []
    avg_iterations = []
    avg_times = []
    
    for m in metrics_data:
        if m and 'success_count' in m and 'total_samples' in m:
            success_rates.append(m['success_count'] / m['total_samples'] * 100)
        else:
            success_rates.append(0)
        
        if m and 'avg_iterations' in m:
            avg_iterations.append(m['avg_iterations'])
        else:
            avg_iterations.append(0)
        
        if m and 'avg_time_ms' in m and m['avg_time_ms'] > 0:
            avg_times.append(m['avg_time_ms'])
        else:
            avg_times.append(0)
    
    # Plot 1: Success Rate
    ax = axes[0]
    bars = ax.bar(decoders, success_rates, color=COLOR_PALETTE, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=11)
    ax.set_title('Success Rate', fontweight='bold', fontsize=12, pad=10)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    for bar, rate in zip(bars, success_rates):
        if rate > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{rate:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    # Plot 2: Average Iterations
    ax = axes[1]
    bars = ax.bar(decoders, avg_iterations, color=COLOR_PALETTE, alpha=0.8,
                 edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Iterations', fontweight='bold', fontsize=11)
    ax.set_title('Iterations', fontweight='bold', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    max_iters = max(avg_iterations) if avg_iterations else 1
    for bar, iters in zip(bars, avg_iterations):
        if iters > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_iters * 0.02,
                   f'{iters:.1f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    # Plot 3: Timing (if available) or show message
    ax = axes[2]
    if any(t > 0 for t in avg_times):
        bars = ax.bar(decoders, avg_times, color=COLOR_PALETTE, alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Average Time (ms)', fontweight='bold', fontsize=11)
        ax.set_title('Decoding Time', fontweight='bold', fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        max_time = max(avg_times) if avg_times else 1
        for bar, time_ms in zip(bars, avg_times):
            if time_ms > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_time * 0.02,
                       f'{time_ms:.0f}ms', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    else:
        # Show iterations again if no timing data
        bars = ax.bar(decoders, avg_iterations, color=COLOR_PALETTE, alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Average Iterations', fontweight='bold', fontsize=11)
        ax.set_title('Iterations (duplicate)', fontweight='bold', fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        max_iters = max(avg_iterations) if avg_iterations else 1
        for bar, iters in zip(bars, avg_iterations):
            if iters > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_iters * 0.02,
                       f'{iters:.1f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'decoding_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'decoding_performance.png'}")


def visualize_bb_code_structure(data: Dict, output_dir: Path) -> None:
    """
    Visualize the bivariate bicycle code structure.
    
    Shows representative rounds and partition structure for the code.
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save visualization
    """
    code_info = data.get('code_info', {})
    
    if not code_info:
        print("No code information found in benchmark output")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Code Structure & Partitioning', fontsize=16, fontweight='bold', y=0.98)
    
    # Extract code parameters
    num_detectors = code_info.get('num_detectors', 2160)
    num_variables = code_info.get('num_variables', 87012)
    detectors_per_round = code_info.get('detectors_per_round', 72)
    total_rounds = code_info.get('total_rounds', 30)
    
    # Extract partitioning info
    partition_analyses = data.get('partition_analyses', [])
    rounds_per_partition = 10  # Default
    for analysis in partition_analyses:
        if analysis['strategy'].startswith('Time-axis'):
            match = re.search(r'(\d+)', analysis['strategy'])
            if match:
                rounds_per_partition = int(match.group(1))
                break
    
    num_partitions = (total_rounds + rounds_per_partition - 1) // rounds_per_partition
    
    # Plot 1: Show only first 6 rounds (representative sample)
    ax1 = axes[0]
    rounds_to_show = min(6, total_rounds)
    rounds_indices = list(range(rounds_to_show))
    
    round_height = 0.12
    round_spacing = 0.02
    y_start = 0.05
    
    colors = plt.cm.viridis(np.linspace(0, 0.7, rounds_to_show))
    
    for idx, round_idx in enumerate(rounds_indices):
        y_pos = y_start + (rounds_to_show - 1 - idx) * (round_height + round_spacing)
        
        rect = mpatches.Rectangle((0, y_pos), 1, round_height, 
                                 facecolor=colors[idx], 
                                 edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.add_patch(rect)
        
        ax1.text(0.5, y_pos + round_height/2, f'Round {round_idx}\n{detectors_per_round} detectors',
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    if total_rounds > rounds_to_show:
        # Add indicator for remaining rounds
        y_pos = y_start - round_spacing - round_height
        ax1.text(0.5, y_pos, f'... ({total_rounds - rounds_to_show} more rounds)',
                ha='center', va='top', fontsize=9, style='italic', color='gray')
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(0, rounds_to_show * (round_height + round_spacing) + y_start + 0.1)
    ax1.set_title(f'Rounds (first {rounds_to_show} of {total_rounds})', 
                 fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Plot 2: Show only first 3 partitions (representative sample)
    ax2 = axes[1]
    partitions_to_show = min(3, num_partitions)
    partition_indices = list(range(partitions_to_show))
    
    partition_colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    partition_height = 0.25
    partition_spacing = 0.03
    
    y_pos = 0.05
    for idx, part_idx in enumerate(partition_indices):
        start_round = part_idx * rounds_per_partition
        end_round = min(start_round + rounds_per_partition, total_rounds)
        num_rounds_in_part = end_round - start_round
        
        rect = mpatches.Rectangle((0, y_pos), 1, partition_height,
                                 facecolor=partition_colors[part_idx % len(partition_colors)],
                                 edgecolor='black', linewidth=2.5, alpha=0.8)
        ax2.add_patch(rect)
        
        label = f'P{part_idx}: R{start_round}-{end_round-1}\n{num_rounds_in_part * detectors_per_round} dets'
        ax2.text(0.5, y_pos + partition_height/2, label,
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        y_pos += partition_height + partition_spacing
    
    if num_partitions > partitions_to_show:
        # Add indicator for remaining partitions
        y_pos += partition_spacing
        ax2.text(0.5, y_pos, f'... ({num_partitions - partitions_to_show} more)',
                ha='center', va='bottom', fontsize=9, style='italic', color='gray')
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(0, y_pos + 0.1)
    title = f'Partitions (first {partitions_to_show} of {num_partitions})\n{rounds_per_partition} rounds/partition'
    ax2.set_title(title, fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'bb_code_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'bb_code_structure.png'}")


def visualize_partition_structure(data: Dict, output_dir: Path) -> None:
    """
    Visualize partition structure using the check matrix.
    
    Shows how detectors and variables are partitioned across time rounds.
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save visualization
    """
    
    code_info = data.get('code_info', {})
    if not code_info:
        print("No code information found - cannot load check matrix")
        return
    
    # Locate the check matrix file
    # The data files are in crates/relay_bp/data/
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    # Determine which round number to use
    total_rounds = code_info.get('total_rounds', 30)
    matrix_prefix = f"72_12_6_r{total_rounds}"
    matrix_path = data_dir / f"{matrix_prefix}_Hdec.npz"
    
    if not matrix_path.exists():
        # Look for matching file with different round count
        r30_files = list(data_dir.glob("72_12_6_r30_Hdec.npz"))
        if r30_files:
            matrix_path = r30_files[0]
        else:
            # Create error visualization
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.text(0.5, 0.5, f'Partition Structure Visualization\n\n'
                   f'Check matrix file not found at:\n{matrix_path}\n\n'
                   f'Expected location: crates/relay_bp/data/{matrix_prefix}_Hdec.npz',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, 
                            edgecolor='black', linewidth=1.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Partition Structure Visualization\n(Check matrix file not found)', 
                        fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(output_dir / 'partition_structure.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_dir / 'partition_structure.png'}")
            return
    
    # Load the check matrix
    try:
        H = load_npz(str(matrix_path))
        print(f"Loaded check matrix: {H.shape} with {H.nnz} non-zeros")
    except Exception as e:
        print(f"Error loading check matrix: {e}")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Partition Structure Visualization', fontsize=16, fontweight='bold', y=0.98)
    
    num_detectors, num_variables = H.shape
    detectors_per_round = code_info.get('detectors_per_round', 72)
    
    # Plot 1: Sparse matrix pattern with round boundaries
    ax1 = axes[0]
    
    # Sample the matrix for visualization (too large to show all)
    sample_detectors = min(500, num_detectors)
    sample_variables = min(2000, num_variables)
    
    # Get a sample of the matrix
    H_sample = H[:sample_detectors, :sample_variables].toarray()
    
    # Create heatmap
    im1 = ax1.imshow(H_sample, aspect='auto', cmap='YlOrRd', interpolation='nearest', 
                    vmin=0, vmax=1)
    ax1.set_xlabel('Error Variables (sampled)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Detectors (sampled)', fontweight='bold', fontsize=11)
    ax1.set_title(f'Check Matrix Structure (Sample)\n{num_detectors} × {num_variables} total',
                 fontweight='bold', fontsize=12, pad=10)
    
    # Add round boundaries
    if detectors_per_round > 0:
        for round_idx in range(1, sample_detectors // detectors_per_round + 1):
            y_boundary = round_idx * detectors_per_round
            if y_boundary < sample_detectors:
                ax1.axhline(y=y_boundary, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.colorbar(im1, ax=ax1, label='Connection (1=connected)')
    
    # Plot 2: Partition visualization (conceptual based on rounds)
    ax2 = axes[1]
    
    # Get partition info from data
    partition_analyses = data.get('partition_analyses', [])
    rounds_per_partition = None
    for analysis in partition_analyses:
        if analysis['strategy'].startswith('Time-axis'):
            match = re.search(r'(\d+)', analysis['strategy'])
            if match:
                rounds_per_partition = int(match.group(1))
                break
    
    if rounds_per_partition is None:
        rounds_per_partition = 10  # Default
    
    total_rounds = num_detectors // detectors_per_round
    num_partitions = (total_rounds + rounds_per_partition - 1) // rounds_per_partition
    
    # Create a conceptual partition visualization
    partition_colors = plt.cm.Set3(np.linspace(0, 1, max(num_partitions, 6)))
    
    # Draw partitions as blocks
    y_pos = 0
    partition_height = 0.15
    partition_spacing = 0.02
    
    for part_idx in range(min(num_partitions, 10)):  # Show up to 10 partitions
        start_round = part_idx * rounds_per_partition
        end_round = min(start_round + rounds_per_partition, total_rounds)
        num_rounds_in_part = end_round - start_round
        
        # Draw partition block
        rect = mpatches.Rectangle((0, y_pos), 1, partition_height,
                                 facecolor=partition_colors[part_idx % len(partition_colors)],
                                 edgecolor='black', linewidth=2, alpha=0.7)
        ax2.add_patch(rect)
        
        # Label
        label = f'Partition {part_idx}\nRounds {start_round}-{end_round-1}\n'
        label += f'{num_rounds_in_part * detectors_per_round} detectors'
        ax2.text(0.5, y_pos + partition_height/2, label,
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        y_pos += partition_height + partition_spacing
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(0, y_pos + 0.05)
    ax2.set_title(f'Time-Axis Partitioning\n{rounds_per_partition} rounds/partition → {num_partitions} partitions',
                 fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'partition_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'partition_structure.png'}")


def visualize_decoding_agreement(data: Dict, output_dir: Path) -> None:
    """
    Visualize agreement between different decoders.
    
    Shows how often fusion decoders produce identical results compared to baseline.
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save visualization
    """
    agreement = data.get('agreement', {})
    
    if not agreement:
        print("  No agreement data found - skipping")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Decoding Agreement Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Extract agreement data
    time_axis_agree = agreement.get('time_axis_vs_baseline', {})
    count_based_agree = agreement.get('count_based_vs_baseline', {})
    
    # Plot 1: Agreement percentages
    ax1 = axes[0]
    comparisons = []
    identical_counts = []
    total_counts = []
    percentages = []
    
    if time_axis_agree:
        comparisons.append('Time-axis\nvs Baseline')
        identical_counts.append(time_axis_agree.get('identical', 0))
        total_counts.append(time_axis_agree.get('total', 1))
        percentages.append(time_axis_agree.get('identical', 0) / time_axis_agree.get('total', 1) * 100)
    
    if count_based_agree:
        comparisons.append('Count-based\nvs Baseline')
        identical_counts.append(count_based_agree.get('identical', 0))
        total_counts.append(count_based_agree.get('total', 1))
        percentages.append(count_based_agree.get('identical', 0) / count_based_agree.get('total', 1) * 100)
    
    if comparisons:
        bars = ax1.bar(comparisons, percentages, color=[COLOR_TIME_AXIS, COLOR_COUNT_BASED][:len(comparisons)],
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Agreement Rate (%)', fontweight='bold', fontsize=11)
        ax1.set_title('Decoder Agreement with Baseline', fontweight='bold', fontsize=12, pad=10)
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_axisbelow(True)
        
        # Add value labels
        for bar, pct, identical, total in zip(bars, percentages, identical_counts, total_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{pct:.1f}%\n({identical}/{total})', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    # Plot 2: Agreement breakdown
    ax2 = axes[1]
    if comparisons:
        # Create a simple breakdown showing agreement vs disagreement
        agree_data = []
        disagree_data = []
        labels = []
        
        if time_axis_agree:
            total = time_axis_agree.get('total', 1)
            identical = time_axis_agree.get('identical', 0)
            agree_data.append(identical)
            disagree_data.append(total - identical)
            labels.append('Time-axis')
        
        if count_based_agree:
            total = count_based_agree.get('total', 1)
            identical = count_based_agree.get('identical', 0)
            agree_data.append(identical)
            disagree_data.append(total - identical)
            labels.append('Count-based')
        
        if labels:
            x = np.arange(len(labels))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, agree_data, width, label='Agree', 
                           color=COLOR_TIME_AXIS, alpha=0.8, edgecolor='black', linewidth=1.5)
            bars2 = ax2.bar(x + width/2, disagree_data, width, label='Disagree',
                          color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=1.5)
            
            ax2.set_ylabel('Number of Samples', fontweight='bold', fontsize=11)
            ax2.set_title('Agreement Breakdown', fontweight='bold', fontsize=12, pad=10)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax2.set_axisbelow(True)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'{int(height)}', ha='center', va='bottom',
                                fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'decoding_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'decoding_agreement.png'}")


def visualize_sparse_matrix_statistics(data: Dict, output_dir: Path) -> None:
    """
    Visualize sparse matrix statistics.
    
    Shows degree distributions for detectors and variables, average degrees per round,
    and overall matrix sparsity statistics.
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save visualization
    """
    code_info = data.get('code_info', {})
    
    # Load check matrix
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    total_rounds = code_info.get('total_rounds', 30)
    matrix_prefix = f"72_12_6_r{total_rounds}"
    matrix_path = data_dir / f"{matrix_prefix}_Hdec.npz"
    
    if not matrix_path.exists():
        r30_files = list(data_dir.glob("72_12_6_r30_Hdec.npz"))
        if r30_files:
            matrix_path = r30_files[0]
        else:
            print("Check matrix file not found - skipping sparse matrix statistics")
            return
    
    try:
        H = load_npz(str(matrix_path))
        print(f"Loaded check matrix for statistics: {H.shape} with {H.nnz} non-zeros")
    except Exception as e:
        print(f"Error loading check matrix: {e}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sparse Matrix Statistics', fontsize=16, fontweight='bold', y=0.995)
    
    num_detectors, num_variables = H.shape
    detectors_per_round = code_info.get('detectors_per_round', 72)
    total_rounds = num_detectors // detectors_per_round
    
    # Plot 1: Detector degree distribution
    ax1 = axes[0, 0]
    detector_degrees = np.array(H.sum(axis=1)).flatten()
    ax1.hist(detector_degrees, bins=50, color=COLOR_BASELINE, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Detector Degree (Number of Connections)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax1.set_title(f'Detector Degree Distribution\n(Mean: {detector_degrees.mean():.1f}, Std: {detector_degrees.std():.1f})',
                 fontweight='bold', fontsize=12, pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Plot 2: Variable degree distribution
    ax2 = axes[0, 1]
    variable_degrees = np.array(H.sum(axis=0)).flatten()
    ax2.hist(variable_degrees, bins=50, color=COLOR_TIME_AXIS, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_xlabel('Variable Degree (Number of Connections)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax2.set_title(f'Variable Degree Distribution\n(Mean: {variable_degrees.mean():.1f}, Std: {variable_degrees.std():.1f})',
                 fontweight='bold', fontsize=12, pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Plot 3: Average degree per round
    ax3 = axes[1, 0]
    avg_degrees_per_round = []
    for round_idx in range(total_rounds):
        start_det = round_idx * detectors_per_round
        end_det = (round_idx + 1) * detectors_per_round
        round_matrix = H[start_det:end_det, :]
        avg_degree = round_matrix.sum() / detectors_per_round
        avg_degrees_per_round.append(avg_degree)
    
    ax3.plot(range(total_rounds), avg_degrees_per_round, 'o-', linewidth=2.5,
            markersize=8, color=COLOR_COUNT_BASED, markerfacecolor=COLOR_COUNT_BASED,
            markeredgecolor='white', markeredgewidth=1.5)
    ax3.set_xlabel('Round Index', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Average Detector Degree', fontweight='bold', fontsize=11)
    ax3.set_title('Average Detector Degree per Round', fontweight='bold', fontsize=12, pad=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    
    # Plot 4: Sparsity statistics
    ax4 = axes[1, 1]
    sparsity = 1.0 - (H.nnz / (num_detectors * num_variables))
    density = H.nnz / (num_detectors * num_variables)
    
    stats_text = f"""Matrix Statistics:
    
Total Detectors: {num_detectors:,}
Total Variables: {num_variables:,}
Non-zeros: {H.nnz:,}
Sparsity: {sparsity*100:.2f}%
Density: {density*100:.4f}%

Average Detector Degree: {detector_degrees.mean():.2f}
Average Variable Degree: {variable_degrees.mean():.2f}

Rounds: {total_rounds}
Detectors per Round: {detectors_per_round}
"""
    
    ax4.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
            transform=ax4.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5,
                     edgecolor='black', linewidth=1.5))
    ax4.axis('off')
    ax4.set_title('Matrix Statistics Summary', fontweight='bold', fontsize=12, pad=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_dir / 'sparse_matrix_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'sparse_matrix_statistics.png'}")


def visualize_partition_boundary_heatmap(data: Dict, output_dir: Path) -> None:
    """
    Visualize partition boundaries as a heatmap.
    
    Compares time-axis and count-based partitioning strategies side-by-side,
    showing how boundaries are distributed in detector×variable space.
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save visualization
    """
    code_info = data.get('code_info', {})
    
    # Load check matrix
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    total_rounds = code_info.get('total_rounds', 30)
    matrix_prefix = f"72_12_6_r{total_rounds}"
    matrix_path = data_dir / f"{matrix_prefix}_Hdec.npz"
    
    if not matrix_path.exists():
        r30_files = list(data_dir.glob("72_12_6_r30_Hdec.npz"))
        if r30_files:
            matrix_path = r30_files[0]
        else:
            print("  Check matrix file not found - skipping partition boundary heatmap")
            return
    
    try:
        H = load_npz(str(matrix_path))
    except Exception as e:
        print(f"  Error loading check matrix: {e}")
        return
    
    # Get partition info
    partition_analyses = data.get('partition_analyses', [])
    if not partition_analyses:
        print("  No partition analysis data - skipping partition boundary heatmap")
        return
    
    detectors_per_round = code_info.get('detectors_per_round', 72)
    
    # Find time-axis and count-based strategies with same number of partitions
    time_axis_strategy = None
    count_based_strategy = None
    
    for analysis in partition_analyses:
        if analysis['strategy'].startswith('Time-axis') and not time_axis_strategy:
            time_axis_strategy = analysis
        elif analysis['strategy'].startswith('Count-based') and not count_based_strategy:
            count_based_strategy = analysis
    
    if not time_axis_strategy or not count_based_strategy:
        print("  Need both time-axis and count-based strategies for comparison - skipping")
        return
    
    # Sample matrix for visualization (too large to show all)
    sample_detectors = min(500, H.shape[0])
    sample_variables = min(2000, H.shape[1])
    H_sample = H[:sample_detectors, :sample_variables].toarray()
    
    # Create conceptual partition boundaries
    # For time-axis: partitions by rounds
    rounds_per_partition = None
    match = re.search(r'(\d+)', time_axis_strategy['strategy'])
    if match:
        rounds_per_partition = int(match.group(1))
    
    if rounds_per_partition is None:
        rounds_per_partition = 10
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Partition Boundary Heatmap Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Time-axis partitioning
    ax1 = axes[0]
    boundary_map_time = np.zeros_like(H_sample, dtype=float)
    
    # Mark partition boundaries (conceptual - showing which detector-variable pairs cross boundaries)
    num_partitions_time = time_axis_strategy['num_partitions']
    detectors_per_partition = sample_detectors // num_partitions_time if num_partitions_time > 0 else sample_detectors
    
    for part_idx in range(num_partitions_time):
        start_det = part_idx * detectors_per_partition
        end_det = min((part_idx + 1) * detectors_per_partition, sample_detectors)
        # Variables are shared, so we mark boundaries conceptually
        if part_idx < num_partitions_time - 1:
            # Mark boundary region
            boundary_map_time[end_det-5:end_det+5, :] = 0.5
    
    # Overlay actual connections
    boundary_map_time[H_sample > 0] = 1.0
    
    im1 = ax1.imshow(boundary_map_time, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    ax1.set_xlabel('Error Variables (sampled)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Detectors (sampled)', fontweight='bold', fontsize=11)
    ax1.set_title(f'Time-axis Partitioning\n{time_axis_strategy["strategy"]}\nBoundary Ratio: {time_axis_strategy["avg_boundary_ratio"]*100:.1f}%',
                 fontweight='bold', fontsize=12, pad=10)
    
    # Add partition boundaries
    for part_idx in range(1, num_partitions_time):
        boundary = part_idx * detectors_per_partition
        if boundary < sample_detectors:
            ax1.axhline(y=boundary, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.colorbar(im1, ax=ax1, label='Connection Type')
    
    # Plot 2: Count-based partitioning
    ax2 = axes[1]
    boundary_map_count = np.zeros_like(H_sample, dtype=float)
    
    num_partitions_count = count_based_strategy['num_partitions']
    variables_per_partition = sample_variables // num_partitions_count if num_partitions_count > 0 else sample_variables
    
    # Count-based partitions variables, so boundaries are vertical
    for part_idx in range(num_partitions_count):
        start_var = part_idx * variables_per_partition
        end_var = min((part_idx + 1) * variables_per_partition, sample_variables)
        if part_idx < num_partitions_count - 1:
            # Mark boundary region
            boundary_map_count[:, end_var-10:end_var+10] = 0.5
    
    # Overlay actual connections
    boundary_map_count[H_sample > 0] = 1.0
    
    im2 = ax2.imshow(boundary_map_count, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_xlabel('Error Variables (sampled)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Detectors (sampled)', fontweight='bold', fontsize=11)
    ax2.set_title(f'Count-based Partitioning\n{count_based_strategy["strategy"]}\nBoundary Ratio: {count_based_strategy["avg_boundary_ratio"]*100:.1f}%',
                 fontweight='bold', fontsize=12, pad=10)
    
    # Add partition boundaries
    for part_idx in range(1, num_partitions_count):
        boundary = part_idx * variables_per_partition
        if boundary < sample_variables:
            ax2.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.colorbar(im2, ax=ax2, label='Connection Type')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'partition_boundary_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'partition_boundary_heatmap.png'}")


def visualize_circulant_structure(data: Dict, output_dir: Path) -> None:
    """
    Visualize circulant block structure of the check matrix.
    
    Shows the block-circulant structure of bivariate bicycle codes,
    demonstrating how H = [A | B] where A and B are circulant blocks.
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save visualization
    """
    code_info = data.get('code_info', {})
    
    # Load check matrix
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    total_rounds = code_info.get('total_rounds', 30)
    matrix_prefix = f"72_12_6_r{total_rounds}"
    matrix_path = data_dir / f"{matrix_prefix}_Hdec.npz"
    
    if not matrix_path.exists():
        r30_files = list(data_dir.glob("72_12_6_r30_Hdec.npz"))
        if r30_files:
            matrix_path = r30_files[0]
        else:
            print("  Check matrix file not found - skipping circulant structure visualization")
            return
    
    try:
        H = load_npz(str(matrix_path))
        print(f"  Loaded check matrix for circulant visualization: {H.shape}")
    except Exception as e:
        print(f"  Error loading check matrix: {e}")
        return
    
    # For [[72,12,6]] bivariate bicycle code, blocks are typically 72×72
    block_size = 72
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Circulant Block Structure', fontsize=18, fontweight='bold', y=0.98)
    
    # Plot 1: Single 72×72 block - the key concept
    ax1 = axes[0]
    block_sample = H[:block_size, :block_size].toarray()
    im1 = ax1.imshow(block_sample, aspect='equal', cmap='binary', interpolation='nearest', vmin=0, vmax=1)
    ax1.set_xlabel('Variables (0-71)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Detectors (0-71)', fontweight='bold', fontsize=14)
    ax1.set_title(f'{block_size}×{block_size} Circulant Block\nEach row is a circular shift of row 0', 
                 fontweight='bold', fontsize=15, pad=15)
    
    # Draw block boundaries
    ax1.axhline(y=-0.5, color='blue', linestyle='-', linewidth=3)
    ax1.axhline(y=block_size-0.5, color='blue', linestyle='-', linewidth=3)
    ax1.axvline(x=-0.5, color='blue', linestyle='-', linewidth=3)
    ax1.axvline(x=block_size-0.5, color='blue', linestyle='-', linewidth=3)
    
    plt.colorbar(im1, ax=ax1, label='Connection', shrink=0.8)
    
    # Plot 2: Conceptual diagram
    ax2 = axes[1]
    ax2.text(0.5, 0.7, 'H = [A | B]', ha='center', va='center', fontsize=24, fontweight='bold',
            transform=ax2.transAxes)
    
    # Draw A block
    rect_a = mpatches.Rectangle((0.15, 0.4), 0.3, 0.2, facecolor='#3498db', 
                               edgecolor='black', linewidth=3, alpha=0.8)
    ax2.add_patch(rect_a)
    ax2.text(0.3, 0.5, 'A\n72×72\nCirculant', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    # Draw B block
    rect_b = mpatches.Rectangle((0.55, 0.4), 0.3, 0.2, facecolor='#e74c3c', 
                               edgecolor='black', linewidth=3, alpha=0.8)
    ax2.add_patch(rect_b)
    ax2.text(0.7, 0.5, 'B\n72×72\nCirculant', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    # Add explanation
    ax2.text(0.5, 0.15, 
            'Bivariate Bicycle Code Structure\n\n'
            f'• Each block is {block_size}×{block_size}\n'
            '• Blocks are circulant (rows are circular shifts)\n'
            '• This structure enables efficient belief propagation',
            ha='center', va='center', fontsize=12, fontweight='normal',
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, 
                     edgecolor='black', linewidth=2))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Block Structure Concept', fontweight='bold', fontsize=15, pad=15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'circulant_block_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'circulant_block_structure.png'}")


def visualize_tanner_graph(data: Dict, output_dir: Path) -> None:
    """
    Visualize Tanner graph for a subset of the code.
    
    Creates a bipartite graph showing the connection structure between
    detectors (check nodes) and error variables (variable nodes).
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save visualization
    """
    code_info = data.get('code_info', {})
    
    # Load check matrix
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    total_rounds = code_info.get('total_rounds', 30)
    matrix_prefix = f"72_12_6_r{total_rounds}"
    matrix_path = data_dir / f"{matrix_prefix}_Hdec.npz"
    
    if not matrix_path.exists():
        r30_files = list(data_dir.glob("72_12_6_r30_Hdec.npz"))
        if r30_files:
            matrix_path = r30_files[0]
        else:
            print("  Check matrix file not found - skipping Tanner graph visualization")
            return
    
    try:
        H = load_npz(str(matrix_path))
        print(f"  Loaded check matrix for Tanner graph: {H.shape}")
    except Exception as e:
        print(f"  Error loading check matrix: {e}")
        return
    
    # Sample a small subset for visualization (1-2 rounds)
    detectors_per_round = code_info.get('detectors_per_round', 72)
    rounds_to_sample = 1  # Just one round for clarity
    max_detectors = rounds_to_sample * detectors_per_round
    max_variables = 200  # Limit variables for readability
    
    H_sample = H[:max_detectors, :max_variables]
    
    # Create bipartite graph
    G = nx.Graph()
    
    # Add detector nodes (check nodes)
    detector_nodes = [f'D{i}' for i in range(H_sample.shape[0])]
    G.add_nodes_from(detector_nodes, bipartite=0, node_type='detector')
    
    # Add variable nodes (variable nodes)
    variable_nodes = [f'V{i}' for i in range(H_sample.shape[1])]
    G.add_nodes_from(variable_nodes, bipartite=1, node_type='variable')
    
    # Add edges
    H_coo = H_sample.tocoo()
    for i, j in zip(H_coo.row, H_coo.col):
        G.add_edge(detector_nodes[i], variable_nodes[j])
    
    print(f"  Created Tanner graph with {len(detector_nodes)} detectors, {len(variable_nodes)} variables, {G.number_of_edges()} edges")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('Tanner Graph Visualization', fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Bipartite layout
    ax1 = axes[0]
    pos = nx.bipartite_layout(G, detector_nodes, align='vertical')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, width=0.5, edge_color='gray')
    
    # Draw nodes with different colors
    nx.draw_networkx_nodes(G, pos, nodelist=detector_nodes, ax=ax1,
                          node_color=COLOR_BASELINE, node_size=50, alpha=0.8,
                          node_shape='s', label='Detectors (Check Nodes)')
    nx.draw_networkx_nodes(G, pos, nodelist=variable_nodes, ax=ax1,
                          node_color=COLOR_TIME_AXIS, node_size=30, alpha=0.8,
                          node_shape='o', label='Variables (Variable Nodes)')
    
    # Add labels for a few nodes (to avoid clutter)
    sample_detector_labels = {detector_nodes[i]: f'D{i}' for i in range(0, len(detector_nodes), 10)}
    sample_variable_labels = {variable_nodes[i]: f'V{i}' for i in range(0, len(variable_nodes), 20)}
    nx.draw_networkx_labels(G, pos, labels={**sample_detector_labels, **sample_variable_labels},
                           ax=ax1, font_size=6, font_weight='bold')
    
    ax1.set_title(f'Bipartite Tanner Graph\n({len(detector_nodes)} detectors, {len(variable_nodes)} variables)',
                 fontweight='bold', fontsize=12, pad=10)
    ax1.axis('off')
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Plot 2: Spring layout view
    ax2 = axes[1]
    # Use spring layout for a different perspective
    pos2 = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos2, ax=ax2, alpha=0.2, width=0.3, edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos2, nodelist=detector_nodes, ax=ax2,
                          node_color=COLOR_BASELINE, node_size=30, alpha=0.7,
                          node_shape='s')
    nx.draw_networkx_nodes(G, pos2, nodelist=variable_nodes, ax=ax2,
                          node_color=COLOR_TIME_AXIS, node_size=20, alpha=0.7,
                          node_shape='o')
    
    ax2.set_title('Spring Layout View\n(Alternative perspective)', 
                 fontweight='bold', fontsize=12, pad=10)
    ax2.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'tanner_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'tanner_graph.png'}")


def visualize_boundary_variables(data: Dict, output_dir: Path) -> None:
    """
    Visualize which variables are on partition boundaries.
    
    Shows variables that connect to detectors in multiple partitions,
    which create dependencies between partitions during fusion decoding.
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save visualization
    """
    code_info = data.get('code_info', {})
    partition_analyses = data.get('partition_analyses', [])
    
    if not partition_analyses:
        print("  No partition analysis data - skipping boundary variable visualization")
        return
    
    # Load check matrix
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    total_rounds = code_info.get('total_rounds', 30)
    matrix_prefix = f"72_12_6_r{total_rounds}"
    matrix_path = data_dir / f"{matrix_prefix}_Hdec.npz"
    
    if not matrix_path.exists():
        r30_files = list(data_dir.glob("72_12_6_r30_Hdec.npz"))
        if r30_files:
            matrix_path = r30_files[0]
        else:
            print("  Check matrix file not found - skipping boundary variable visualization")
            return
    
    try:
        H = load_npz(str(matrix_path))
        print(f"  Loaded check matrix for boundary visualization: {H.shape}")
    except Exception as e:
        print(f"  Error loading check matrix: {e}")
        return
    
    # Find time-axis strategy (prefer one with moderate partitions)
    time_axis_strategy = None
    for analysis in partition_analyses:
        if analysis['strategy'].startswith('Time-axis'):
            # Prefer strategies with 2-3 partitions for better visualization
            if 2 <= analysis['num_partitions'] <= 3:
                time_axis_strategy = analysis
                break
    if not time_axis_strategy:
        # Fall back to any time-axis strategy
        for analysis in partition_analyses:
            if analysis['strategy'].startswith('Time-axis'):
                time_axis_strategy = analysis
                break
    
    if not time_axis_strategy:
        print("  No time-axis strategy found - skipping boundary variable visualization")
        return
    
    # Extract rounds per partition
    rounds_per_partition = None
    match = re.search(r'(\d+)', time_axis_strategy['strategy'])
    if match:
        rounds_per_partition = int(match.group(1))
    
    if rounds_per_partition is None:
        rounds_per_partition = 10
    
    detectors_per_round = code_info.get('detectors_per_round', 72)
    num_partitions = time_axis_strategy['num_partitions']
    
    # Show just ONE partition boundary clearly - much simpler
    detectors_per_partition = rounds_per_partition * detectors_per_round
    
    # Sample just around one partition boundary for clarity
    # Show last part of partition 0 and first part of partition 1
    if num_partitions < 2:
        print("  Need at least 2 partitions to show boundary - skipping")
        return
    
    # Sample around the boundary between partition 0 and 1
    boundary_det = detectors_per_partition
    sample_range = min(144, detectors_per_partition)  # Show 2 rounds worth
    start_det = max(0, boundary_det - sample_range // 2)
    end_det = min(H.shape[0], boundary_det + sample_range // 2)
    
    # Limit variables too
    max_variables_to_show = min(300, H.shape[1])
    
    H_sample = H[start_det:end_det, :max_variables_to_show]
    H_sample_dense = H_sample.toarray()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Boundary Variables Visualization', fontsize=18, fontweight='bold', y=0.98)
    
    # Plot 1: Simple visualization showing partition boundary
    ax1 = axes[0]
    
    # Identify boundary variables (connect to both partitions)
    p0_end = boundary_det - start_det
    p1_start = boundary_det - start_det
    
    p0_connections = H_sample_dense[:p0_end, :]
    p1_connections = H_sample_dense[p1_start:, :]
    
    boundary_variable_indices = set()
    for var_idx in range(max_variables_to_show):
        connects_p0 = np.any(p0_connections[:, var_idx] > 0)
        connects_p1 = np.any(p1_connections[:, var_idx] > 0)
        if connects_p0 and connects_p1:
            boundary_variable_indices.add(var_idx)
    
    # Create simple visualization
    boundary_map = np.zeros_like(H_sample_dense, dtype=float)
    for det_idx in range(H_sample_dense.shape[0]):
        for var_idx in range(max_variables_to_show):
            if H_sample_dense[det_idx, var_idx] > 0:
                if var_idx in boundary_variable_indices:
                    boundary_map[det_idx, var_idx] = 1.0  # Boundary variable - bright
                else:
                    boundary_map[det_idx, var_idx] = 0.3  # Regular connection - dim
    
    from matplotlib import cm
    cmap = cm.get_cmap('RdYlGn')
    im1 = ax1.imshow(boundary_map, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    ax1.set_xlabel('Error Variables', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Detectors', fontweight='bold', fontsize=14)
    ax1.set_title(f'Partition Boundary\n{time_axis_strategy["strategy"]}\n\nYellow/Red = Boundary Variables\n(Green = Regular Connections)',
                 fontweight='bold', fontsize=15, pad=15)
    
    # Draw partition boundary line
    boundary_line = p0_end
    ax1.axhline(y=boundary_line, color='blue', linestyle='-', linewidth=4, alpha=0.9, label='Partition Boundary')
    
    # Add labels
    ax1.text(0.02, boundary_line/2, 'Partition 0', ha='left', va='center', 
            fontsize=12, fontweight='bold', transform=ax1.get_yaxis_transform(),
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax1.text(0.02, boundary_line + (H_sample_dense.shape[0] - boundary_line)/2, 'Partition 1', 
            ha='left', va='center', fontsize=12, fontweight='bold', 
            transform=ax1.get_yaxis_transform(),
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(0.3), label='Regular Connection'),
        Patch(facecolor=cmap(1.0), label='Boundary Variable'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.colorbar(im1, ax=ax1, label='Connection Type', shrink=0.8)
    
    # Plot 2: Summary and explanation
    ax2 = axes[1]
    num_boundary_vars = len(boundary_variable_indices)
    total_vars_in_sample = max_variables_to_show
    
    explanation_text = f"""Boundary Variables Explained

What are boundary variables?
• Variables that connect to detectors in
  MULTIPLE partitions
• These create dependencies between partitions
• Fewer boundary variables = better partitioning

Results for {time_axis_strategy['strategy']}:
• Partitions: {num_partitions}
• Rounds per partition: {rounds_per_partition}
• Boundary variables in sample: {num_boundary_vars}
• Boundary ratio: {time_axis_strategy['avg_boundary_ratio']*100:.1f}%

Why this matters:
Lower boundary ratio means partitions are
more independent, enabling better parallel
decoding and faster convergence.
"""
    
    ax2.text(0.5, 0.5, explanation_text, ha='center', va='center', fontsize=12,
            transform=ax2.transAxes, fontweight='normal',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5,
                     edgecolor='black', linewidth=2))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Key Concept', fontweight='bold', fontsize=15, pad=15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'boundary_variables.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'boundary_variables.png'}")


def create_summary_report(data: Dict, output_dir: Path) -> None:
    """
    Create a text summary report of all metrics.
    
    Args:
        data: Parsed benchmark data dictionary
        output_dir: Directory to save report
    """
    report_path = output_dir / 'visualization_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TIME-AXIS PARTITIONING VISUALIZATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PARTITION QUALITY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        for analysis in data.get('partition_analyses', []):
            f.write(f"\nStrategy: {analysis['strategy']}\n")
            f.write(f"  Partitions: {analysis['num_partitions']}\n")
            f.write(f"  Avg Boundary Ratio: {analysis['avg_boundary_ratio']*100:.2f}%\n")
            f.write(f"  Min Boundary Ratio: {analysis['min_boundary_ratio']*100:.2f}%\n")
            f.write(f"  Max Boundary Ratio: {analysis['max_boundary_ratio']*100:.2f}%\n")
        
        f.write("\n\nDECODING PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        
        for name, metrics in [('Baseline', data.get('baseline_metrics', {})),
                             ('Time-axis Fusion', data.get('time_axis_metrics', {})),
                             ('Count-based Fusion', data.get('count_based_metrics', {}))]:
            if metrics:
                f.write(f"\n{name}:\n")
                if 'success_count' in metrics:
                    f.write(f"  Success Rate: {metrics['success_count']}/{metrics.get('total_samples', 1)} "
                           f"({metrics['success_count']/metrics.get('total_samples', 1)*100:.1f}%)\n")
                if 'syndrome_satisfaction' in metrics:
                    f.write(f"  Syndrome Satisfaction: {metrics['syndrome_satisfaction']}/{metrics.get('total_samples', 1)} "
                           f"({metrics['syndrome_satisfaction']/metrics.get('total_samples', 1)*100:.1f}%)\n")
                if 'avg_iterations' in metrics:
                    f.write(f"  Avg Iterations: {metrics['avg_iterations']:.1f}\n")
                if 'avg_time_ms' in metrics:
                    f.write(f"  Avg Time: {metrics['avg_time_ms']:.2f} ms\n")
    
    print(f"Saved: {report_path}")


def main():
    """Main function to run visualizations."""
    parser = argparse.ArgumentParser(
        description='Visualize time-axis partitioning fusion decoder results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark and create visualizations
  python visualize_fusion.py

  # Use saved benchmark output (skip running benchmark)
  python visualize_fusion.py --no-run

  # Specify custom output directory
  python visualize_fusion.py --output-dir ./my_visualizations
        """
    )
    parser.add_argument(
        '--no-run',
        action='store_true',
        help='Skip running the benchmark, use saved output if available'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: visualizations/)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("FUSION DECODER VISUALIZATION")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Run benchmark and parse output
    if not args.no_run:
        try:
            print("Step 1: Running benchmark...")
            output = run_benchmark()
            
            # Save raw output
            with open(output_dir / 'benchmark_output.txt', 'w') as f:
                f.write(output)
            
            print("Step 2: Parsing benchmark output...")
            data = parse_benchmark_output(output)
            
            # Save parsed data as JSON
            with open(output_dir / 'parsed_data.json', 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            print(f"\nError running benchmark: {e}")
            print("Trying to use saved benchmark output if available...")
            args.no_run = True  # Fall back to saved output
    
    # Load data (either from just-run benchmark or saved output)
    saved_output = output_dir / 'benchmark_output.txt'
    json_path = output_dir / 'parsed_data.json'
    
    if saved_output.exists():
        print("Loading benchmark output...")
        with open(saved_output, 'r') as f:
            output = f.read()
        
        if output.strip():  # Only parse if file has content
            data = parse_benchmark_output(output)
        else:
            print("  Benchmark output file is empty, trying parsed_data.json...")
            data = {}
    
        # Load from parsed_data.json and merge with any parsed data
        if json_path.exists():
            with open(json_path, 'r') as f:
                parsed_data = json.load(f)
            # Merge parsed data (it may be more complete)
            for key in parsed_data:
                if key not in data or not data[key]:
                    data[key] = parsed_data[key]
        
        # Save/update parsed data
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif json_path.exists():
        print("Loading from parsed_data.json...")
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        print("Error: No benchmark output found.")
        print("\nPlease run the benchmark first:")
        print("  cargo run --example benchmark_time_axis_partitioning --release")
        print("Or run this script without --no-run flag")
        sys.exit(1)
    
    # Create visualizations
    try:
        print("Step 3: Creating visualizations...")
        visualize_partition_quality(data, output_dir)
        visualize_decoding_performance(data, output_dir)
        visualize_bb_code_structure(data, output_dir)
        visualize_partition_structure(data, output_dir)
        
        print("\nStep 4: Creating additional visualizations...")
        visualize_decoding_agreement(data, output_dir)
        visualize_sparse_matrix_statistics(data, output_dir)
        visualize_partition_boundary_heatmap(data, output_dir)
        visualize_circulant_structure(data, output_dir)
        visualize_tanner_graph(data, output_dir)
        visualize_boundary_variables(data, output_dir)
        
        print("Step 5: Creating summary report...")
        create_summary_report(data, output_dir)
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETE")
        print("=" * 80)
        print(f"\nAll visualizations saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - partition_quality.png")
        print("  - decoding_performance.png")
        print("  - bb_code_structure.png")
        print("  - partition_structure.png")
        print("  - decoding_agreement.png")
        print("  - sparse_matrix_statistics.png")
        print("  - partition_boundary_heatmap.png")
        print("  - circulant_block_structure.png")
        print("  - tanner_graph.png")
        print("  - boundary_variables.png")
        print("  - visualization_report.txt")
        print("  - benchmark_output.txt")
        print("  - parsed_data.json")
        
    except Exception as e:
        print(f"\nError creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

