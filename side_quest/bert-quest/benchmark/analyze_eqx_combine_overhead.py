#!/usr/bin/env python3
"""
Focused trace analysis to find eqx.combine() overhead in train_step.
Looks specifically at Python function calls within the train_step operation.
"""

import gzip
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def extract_train_step_details(trace_path, implementation_name):
    """Extract detailed timing breakdown of train_step function calls."""
    
    with gzip.open(trace_path, 'rt') as f:
        trace_data = json.load(f)
    
    trace_events = trace_data.get('traceEvents', [])
    
    # Find all train_step related events
    train_step_events = []
    for event in trace_events:
        name = event.get('name', '')
        if 'train_step' in name.lower() or 'gradient' in name.lower():
            train_step_events.append(event)
    
    # Group by step number (if available)
    steps = defaultdict(list)
    
    for event in train_step_events:
        name = event.get('name', '')
        dur = event.get('dur', 0)
        ts = event.get('ts', 0)
        phase = event.get('ph', '')
        
        # Try to extract step number from timestamp or context
        steps['all'].append({
            'name': name,
            'dur_us': dur,
            'dur_ms': dur / 1000,
            'ts': ts,
            'phase': phase,
            'args': event.get('args', {})
        })
    
    return steps


def find_equinox_operations(trace_path):
    """Find all Equinox-related operations in the trace."""
    
    with gzip.open(trace_path, 'rt') as f:
        trace_data = json.load(f)
    
    trace_events = trace_data.get('traceEvents', [])
    
    eqx_ops = []
    combine_ops = []
    
    for event in trace_events:
        name = event.get('name', '')
        dur = event.get('dur', 0)
        
        # Look for equinox module operations
        if 'equinox' in name.lower() or 'eqx' in name.lower():
            eqx_ops.append({
                'name': name,
                'dur_ms': dur / 1000,
                'phase': event.get('ph', ''),
                'cat': event.get('cat', '')
            })
        
        # Look for combine operations specifically
        if 'combine' in name.lower():
            combine_ops.append({
                'name': name,
                'dur_ms': dur / 1000,
                'phase': event.get('ph', ''),
                'cat': event.get('cat', '')
            })
    
    return eqx_ops, combine_ops


def analyze_python_function_calls(trace_path):
    """Extract Python function call timings."""
    
    with gzip.open(trace_path, 'rt') as f:
        trace_data = json.load(f)
    
    trace_events = trace_data.get('traceEvents', [])
    
    # Look for Python function calls
    python_calls = defaultdict(list)
    
    for event in trace_events:
        name = event.get('name', '')
        cat = event.get('cat', '')
        dur = event.get('dur', 0)
        
        # Python function calls usually have specific categories
        if cat in ['python', 'Python', 'function', 'Function']:
            python_calls[name].append(dur / 1000)  # Convert to ms
        
        # Also look for named Python operations
        if any(keyword in name.lower() for keyword in ['partition', 'combine', 'filter', 'tree']):
            python_calls[name].append(dur / 1000)
    
    return python_calls


def compare_implementations(manual_jit_path, flax_path):
    """Compare the two implementations to find overhead differences."""
    
    print("=" * 80)
    print("FOCUSED ANALYSIS: eqx.combine() Overhead")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("1. TRAIN STEP BREAKDOWN")
    print("=" * 80)
    
    manual_steps = extract_train_step_details(manual_jit_path, "Manual JIT")
    flax_steps = extract_train_step_details(flax_path, "Flax")
    
    print("\nManual JIT train_step events:")
    manual_sorted = sorted(manual_steps['all'], key=lambda x: x['dur_ms'], reverse=True)[:30]
    for i, event in enumerate(manual_sorted, 1):
        if event['dur_ms'] > 0.01:  # Filter out very short events
            print(f"  {i:2d}. {event['name'][:70]:70s} {event['dur_ms']:>10.2f} ms")
    
    print("\nFlax train_step events:")
    flax_sorted = sorted(flax_steps['all'], key=lambda x: x['dur_ms'], reverse=True)[:30]
    for i, event in enumerate(flax_sorted, 1):
        if event['dur_ms'] > 0.01:
            print(f"  {i:2d}. {event['name'][:70]:70s} {event['dur_ms']:>10.2f} ms")
    
    print("\n" + "=" * 80)
    print("2. EQUINOX OPERATIONS")
    print("=" * 80)
    
    manual_eqx, manual_combine = find_equinox_operations(manual_jit_path)
    flax_eqx, flax_combine = find_equinox_operations(flax_path)
    
    print(f"\nManual JIT:")
    print(f"  Total Equinox operations: {len(manual_eqx)}")
    print(f"  Total combine operations: {len(manual_combine)}")
    
    if manual_eqx:
        print("\n  Equinox operations:")
        for op in manual_eqx[:20]:
            print(f"    {op['name'][:65]:65s} {op['dur_ms']:>10.2f} ms")
    
    if manual_combine:
        total_combine_time = sum(op['dur_ms'] for op in manual_combine)
        print(f"\n  Combine operations (TOTAL: {total_combine_time:.2f} ms):")
        for op in manual_combine:
            print(f"    {op['name'][:65]:65s} {op['dur_ms']:>10.2f} ms")
    
    print(f"\nFlax:")
    print(f"  Total Equinox operations: {len(flax_eqx)}")
    print(f"  Total combine operations: {len(flax_combine)}")
    
    if flax_eqx:
        print("\n  Equinox operations:")
        for op in flax_eqx[:20]:
            print(f"    {op['name'][:65]:65s} {op['dur_ms']:>10.2f} ms")
    
    print("\n" + "=" * 80)
    print("3. PYTHON FUNCTION CALLS")
    print("=" * 80)
    
    manual_python = analyze_python_function_calls(manual_jit_path)
    flax_python = analyze_python_function_calls(flax_path)
    
    print("\nManual JIT Python operations:")
    if manual_python:
        for name, times in sorted(manual_python.items(), key=lambda x: sum(x[1]), reverse=True)[:20]:
            total_time = sum(times)
            count = len(times)
            avg_time = total_time / count if count > 0 else 0
            print(f"  {name[:60]:60s} {total_time:>10.2f} ms ({count} calls, avg {avg_time:.2f} ms)")
    else:
        print("  No Python operations found in trace")
    
    print("\nFlax Python operations:")
    if flax_python:
        for name, times in sorted(flax_python.items(), key=lambda x: sum(x[1]), reverse=True)[:20]:
            total_time = sum(times)
            count = len(times)
            avg_time = total_time / count if count > 0 else 0
            print(f"  {name[:60]:60s} {total_time:>10.2f} ms ({count} calls, avg {avg_time:.2f} ms)")
    else:
        print("  No Python operations found in trace")
    
    print("\n" + "=" * 80)
    print("4. SUMMARY OF OVERHEAD")
    print("=" * 80)
    
    # Calculate differences
    if manual_combine:
        total_combine = sum(op['dur_ms'] for op in manual_combine)
        print(f"\neqx.combine() total overhead in Manual JIT: {total_combine:.2f} ms")
    else:
        print("\nNo eqx.combine() operations found in Manual JIT trace")
        print("This might mean:")
        print("  1. The operation is too fast to be captured (< 1μs)")
        print("  2. It's compiled away by JAX")
        print("  3. It's not labeled explicitly in the trace")
    
    # Check for PyTree operations
    print("\nLooking for PyTree operations in both traces...")
    
    manual_tree_ops = {name: times for name, times in manual_python.items() 
                      if any(kw in name.lower() for kw in ['tree', 'pytree', 'flatten', 'unflatten'])}
    flax_tree_ops = {name: times for name, times in flax_python.items() 
                    if any(kw in name.lower() for kw in ['tree', 'pytree', 'flatten', 'unflatten'])}
    
    if manual_tree_ops:
        print("\nManual JIT PyTree operations:")
        for name, times in manual_tree_ops.items():
            print(f"  {name}: {sum(times):.2f} ms total")
    
    if flax_tree_ops:
        print("\nFlax PyTree operations:")
        for name, times in flax_tree_ops.items():
            print(f"  {name}: {sum(times):.2f} ms total")


if __name__ == "__main__":
    # Use the newer traces from the session summary
    manual_jit_trace = Path("/mnt/carles/fme/trace_manual_jit/plugins/profile/2025_10_16_02_01_36/t1v-n-42336699-w-0.trace.json.gz")
    flax_trace = Path("/mnt/carles/fme/trace_flax_no_dropout/plugins/profile/2025_10_15_10_11_47/t1v-n-42336699-w-0.trace.json.gz")
    
    if not manual_jit_trace.exists():
        print(f"ERROR: Manual JIT trace not found at {manual_jit_trace}")
        # Try older trace
        manual_jit_trace = Path("/mnt/carles/fme/trace_manual_jit/plugins/profile/2025_10_15_05_52_01/t1v-n-42336699-w-0.trace.json.gz")
        if not manual_jit_trace.exists():
            print(f"ERROR: Older trace also not found")
            exit(1)
        else:
            print(f"Using older trace: {manual_jit_trace}")
    
    if not flax_trace.exists():
        print(f"ERROR: Flax trace not found at {flax_trace}")
        exit(1)
    
    compare_implementations(manual_jit_trace, flax_trace)
