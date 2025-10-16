#!/usr/bin/env python3
"""
Compare profiler traces between manual JIT and Flax implementations.
Analyzes XLA operations, kernel execution times, and memory usage.
"""

import gzip
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def analyze_trace_json(trace_path):
    """Analyze a trace.json.gz file and extract timing information."""
    
    with gzip.open(trace_path, 'rt') as f:
        trace_data = json.load(f)
    
    stats = {
        'total_events': 0,
        'categories': defaultdict(int),
        'phases': defaultdict(int),
        'op_durations': defaultdict(list),
        'custom_annotations': [],
        'xla_ops': defaultdict(list),
        'total_duration_us': 0,
        'gradient_compute_time': 0,
        'weight_update_time': 0,
        'train_step_times': [],
    }
    
    trace_events = trace_data.get('traceEvents', [])
    
    for event in trace_events:
        stats['total_events'] += 1
        
        cat = event.get('cat', 'unknown')
        phase = event.get('ph', 'unknown')
        name = event.get('name', '')
        dur = event.get('dur', 0)
        
        stats['categories'][cat] += 1
        stats['phases'][phase] += 1
        
        if dur > 0:
            stats['op_durations'][name].append(dur)
            stats['total_duration_us'] += dur
        
        if 'gradient_compute' in name.lower():
            stats['gradient_compute_time'] += dur
        
        if 'weight_update' in name.lower():
            stats['weight_update_time'] += dur
        
        if 'train_step' in name.lower() and phase == 'X':
            stats['train_step_times'].append(dur)
        
        if cat == 'XLA' or 'xla' in name.lower():
            stats['xla_ops'][name].append(dur)
        
        if cat == 'custom' or 'Annotation' in name:
            stats['custom_annotations'].append({
                'name': name,
                'dur': dur,
                'phase': phase
            })
    
    return stats


def compare_traces(manual_jit_path, flax_path):
    """Compare two trace files and print the differences."""
    
    print("=" * 80)
    print("PROFILER TRACE COMPARISON: Manual JIT vs Flax (No Dropout)")
    print("=" * 80)
    
    print("\nAnalyzing Manual JIT trace...")
    manual_stats = analyze_trace_json(manual_jit_path)
    
    print("Analyzing Flax trace...")
    flax_stats = analyze_trace_json(flax_path)
    
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal Events:")
    print(f"  Manual JIT: {manual_stats['total_events']:,}")
    print(f"  Flax:       {flax_stats['total_events']:,}")
    
    print(f"\nTotal Duration (ms):")
    print(f"  Manual JIT: {manual_stats['total_duration_us'] / 1000:.2f}")
    print(f"  Flax:       {flax_stats['total_duration_us'] / 1000:.2f}")
    
    print("\n" + "=" * 80)
    print("TRAIN STEP TIMING")
    print("=" * 80)
    
    if manual_stats['train_step_times']:
        manual_step_times = np.array(manual_stats['train_step_times']) / 1000
        print(f"\nManual JIT Train Step Times (ms):")
        print(f"  Mean:   {manual_step_times.mean():.2f}")
        print(f"  Median: {np.median(manual_step_times):.2f}")
        print(f"  Std:    {manual_step_times.std():.2f}")
        print(f"  Min:    {manual_step_times.min():.2f}")
        print(f"  Max:    {manual_step_times.max():.2f}")
    
    if flax_stats['train_step_times']:
        flax_step_times = np.array(flax_stats['train_step_times']) / 1000
        print(f"\nFlax Train Step Times (ms):")
        print(f"  Mean:   {flax_step_times.mean():.2f}")
        print(f"  Median: {np.median(flax_step_times):.2f}")
        print(f"  Std:    {flax_step_times.std():.2f}")
        print(f"  Min:    {flax_step_times.min():.2f}")
        print(f"  Max:    {flax_step_times.max():.2f}")
    
    print("\n" + "=" * 80)
    print("GRADIENT COMPUTE vs WEIGHT UPDATE")
    print("=" * 80)
    
    print(f"\nManual JIT:")
    print(f"  Gradient Compute: {manual_stats['gradient_compute_time'] / 1000:.2f} ms")
    print(f"  Weight Update:    {manual_stats['weight_update_time'] / 1000:.2f} ms")
    
    print(f"\nFlax:")
    print(f"  Gradient Compute: {flax_stats['gradient_compute_time'] / 1000:.2f} ms")
    print(f"  Weight Update:    {flax_stats['weight_update_time'] / 1000:.2f} ms")
    
    print("\n" + "=" * 80)
    print("TOP 20 OPERATIONS BY TOTAL TIME")
    print("=" * 80)
    
    print("\nManual JIT:")
    manual_ops_sorted = sorted(
        [(name, sum(durs)) for name, durs in manual_stats['op_durations'].items()],
        key=lambda x: x[1],
        reverse=True
    )[:20]
    
    for i, (name, total_dur) in enumerate(manual_ops_sorted, 1):
        print(f"  {i:2d}. {name[:60]:60s} {total_dur/1000:>10.2f} ms")
    
    print("\nFlax:")
    flax_ops_sorted = sorted(
        [(name, sum(durs)) for name, durs in flax_stats['op_durations'].items()],
        key=lambda x: x[1],
        reverse=True
    )[:20]
    
    for i, (name, total_dur) in enumerate(flax_ops_sorted, 1):
        print(f"  {i:2d}. {name[:60]:60s} {total_dur/1000:>10.2f} ms")
    
    print("\n" + "=" * 80)
    print("XLA OPERATIONS COMPARISON")
    print("=" * 80)
    
    print(f"\nTotal XLA Operations:")
    print(f"  Manual JIT: {len(manual_stats['xla_ops']):,} unique ops")
    print(f"  Flax:       {len(flax_stats['xla_ops']):,} unique ops")
    
    manual_xla_time = sum(sum(durs) for durs in manual_stats['xla_ops'].values())
    flax_xla_time = sum(sum(durs) for durs in flax_stats['xla_ops'].values())
    
    print(f"\nTotal XLA Time:")
    print(f"  Manual JIT: {manual_xla_time / 1000:.2f} ms")
    print(f"  Flax:       {flax_xla_time / 1000:.2f} ms")
    
    print("\n" + "=" * 80)
    print("EVENT CATEGORIES")
    print("=" * 80)
    
    print("\nManual JIT Categories:")
    for cat, count in sorted(manual_stats['categories'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cat:30s} {count:>10,}")
    
    print("\nFlax Categories:")
    for cat, count in sorted(flax_stats['categories'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cat:30s} {count:>10,}")
    
    print("\n" + "=" * 80)
    print("CUSTOM ANNOTATIONS")
    print("=" * 80)
    
    print(f"\nManual JIT Custom Annotations: {len(manual_stats['custom_annotations'])}")
    for ann in manual_stats['custom_annotations'][:10]:
        print(f"  {ann['name']:40s} {ann['dur']/1000:>10.2f} ms")
    
    print(f"\nFlax Custom Annotations: {len(flax_stats['custom_annotations'])}")
    for ann in flax_stats['custom_annotations'][:10]:
        print(f"  {ann['name']:40s} {ann['dur']/1000:>10.2f} ms")


if __name__ == "__main__":
    manual_jit_trace = Path("/mnt/carles/fme/trace_manual_jit/plugins/profile/2025_10_15_05_52_01/t1v-n-42336699-w-0.trace.json.gz")
    flax_trace = Path("/mnt/carles/fme/trace_flax_no_dropout/plugins/profile/2025_10_15_10_11_47/t1v-n-42336699-w-0.trace.json.gz")
    
    if not manual_jit_trace.exists():
        print(f"ERROR: Manual JIT trace not found at {manual_jit_trace}")
        exit(1)
    
    if not flax_trace.exists():
        print(f"ERROR: Flax trace not found at {flax_trace}")
        exit(1)
    
    compare_traces(manual_jit_trace, flax_trace)
