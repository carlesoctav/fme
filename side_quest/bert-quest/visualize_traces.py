#!/usr/bin/env python3
"""
Visualize and compare profiler traces in detail.
Extracts step-by-step timing information from trace JSON files.
"""

import gzip
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def extract_step_events(trace_path, step_range=(1, 11)):
    """Extract all events for specific training steps."""
    
    with gzip.open(trace_path, 'rt') as f:
        trace_data = json.load(f)
    
    trace_events = trace_data.get('traceEvents', [])
    
    # Group events by step
    step_events = defaultdict(list)
    
    for event in trace_events:
        name = event.get('name', '')
        
        # Look for step annotations
        if 'train_step' in name.lower():
            args = event.get('args', {})
            step_num = args.get('step_num', args.get('step', -1))
            
            # Convert to int if it's a string
            try:
                step_num = int(step_num)
            except (ValueError, TypeError):
                step_num = -1
            
            if step_range[0] <= step_num <= step_range[1]:
                step_events[step_num].append(event)
        
        # Also capture gradient_compute and weight_update
        if 'gradient_compute' in name.lower() or 'weight_update' in name.lower():
            args = event.get('args', {})
            step_num = args.get('step_num', args.get('step', -1))
            
            # Convert to int if it's a string
            try:
                step_num = int(step_num)
            except (ValueError, TypeError):
                step_num = -1
            
            if step_range[0] <= step_num <= step_range[1]:
                step_events[step_num].append(event)
    
    return step_events, trace_events


def analyze_step_breakdown(trace_events, step_num):
    """Analyze a single step's event breakdown."""
    
    breakdown = {
        'train_step': [],
        'gradient_compute': [],
        'weight_update': [],
        'xla_ops': [],
        'other': []
    }
    
    for event in trace_events:
        name = event.get('name', '')
        dur = event.get('dur', 0)
        cat = event.get('cat', '')
        phase = event.get('ph', '')
        
        if phase != 'X':  # Only complete events
            continue
        
        if 'train_step' in name.lower():
            breakdown['train_step'].append({
                'name': name,
                'dur_ms': dur / 1000,
                'cat': cat
            })
        elif 'gradient_compute' in name.lower():
            breakdown['gradient_compute'].append({
                'name': name,
                'dur_ms': dur / 1000,
                'cat': cat
            })
        elif 'weight_update' in name.lower():
            breakdown['weight_update'].append({
                'name': name,
                'dur_ms': dur / 1000,
                'cat': cat
            })
        elif cat == 'XLA' or 'xla' in name.lower():
            breakdown['xla_ops'].append({
                'name': name,
                'dur_ms': dur / 1000,
                'cat': cat
            })
        elif dur > 0:
            breakdown['other'].append({
                'name': name,
                'dur_ms': dur / 1000,
                'cat': cat
            })
    
    return breakdown


def print_trace_comparison(manual_path, flax_path):
    """Print detailed trace comparison."""
    
    print("=" * 100)
    print("DETAILED PROFILER TRACE VISUALIZATION")
    print("=" * 100)
    
    # Load traces
    print("\n[1/4] Loading Manual JIT trace...")
    manual_step_events, manual_all = extract_step_events(manual_path, step_range=(1, 11))
    
    print("[2/4] Loading Flax trace...")
    flax_step_events, flax_all = extract_step_events(flax_path, step_range=(1, 11))
    
    print("[3/4] Analyzing step breakdowns...")
    
    # Analyze each step
    print("\n" + "=" * 100)
    print("STEP-BY-STEP BREAKDOWN (Steps 1-10)")
    print("=" * 100)
    
    for step in range(1, 11):
        print(f"\n{'─' * 100}")
        print(f"STEP {step}")
        print(f"{'─' * 100}")
        
        print(f"\n  Manual JIT:")
        if step in manual_step_events:
            manual_breakdown = analyze_step_breakdown(manual_step_events[step], step)
            
            if manual_breakdown['train_step']:
                total = sum(e['dur_ms'] for e in manual_breakdown['train_step'])
                print(f"    train_step:        {total:>8.2f} ms")
            
            if manual_breakdown['gradient_compute']:
                total = sum(e['dur_ms'] for e in manual_breakdown['gradient_compute'])
                print(f"    gradient_compute:  {total:>8.2f} ms")
            
            if manual_breakdown['weight_update']:
                total = sum(e['dur_ms'] for e in manual_breakdown['weight_update'])
                print(f"    weight_update:     {total:>8.2f} ms")
            
            if manual_breakdown['xla_ops']:
                total = sum(e['dur_ms'] for e in manual_breakdown['xla_ops'])
                print(f"    XLA ops:           {total:>8.2f} ms ({len(manual_breakdown['xla_ops'])} ops)")
        else:
            print(f"    No trace data for step {step}")
        
        print(f"\n  Flax:")
        if step in flax_step_events:
            flax_breakdown = analyze_step_breakdown(flax_step_events[step], step)
            
            if flax_breakdown['train_step']:
                total = sum(e['dur_ms'] for e in flax_breakdown['train_step'])
                print(f"    train_step:        {total:>8.2f} ms")
            
            if flax_breakdown['gradient_compute']:
                total = sum(e['dur_ms'] for e in flax_breakdown['gradient_compute'])
                print(f"    gradient_compute:  {total:>8.2f} ms")
            
            if flax_breakdown['weight_update']:
                total = sum(e['dur_ms'] for e in flax_breakdown['weight_update'])
                print(f"    weight_update:     {total:>8.2f} ms")
            
            if flax_breakdown['xla_ops']:
                total = sum(e['dur_ms'] for e in flax_breakdown['xla_ops'])
                print(f"    XLA ops:           {total:>8.2f} ms ({len(flax_breakdown['xla_ops'])} ops)")
        else:
            print(f"    No trace data for step {step}")
    
    # Overall statistics
    print("\n" + "=" * 100)
    print("OVERALL TRACE STATISTICS")
    print("=" * 100)
    
    print(f"\nManual JIT:")
    print(f"  Total events: {len(manual_all):,}")
    print(f"  Steps captured: {sorted(manual_step_events.keys())}")
    
    print(f"\nFlax:")
    print(f"  Total events: {len(flax_all):,}")
    print(f"  Steps captured: {sorted(flax_step_events.keys())}")
    
    # Event categories
    print("\n" + "=" * 100)
    print("EVENT CATEGORIES (All Events)")
    print("=" * 100)
    
    manual_cats = defaultdict(int)
    for e in manual_all:
        manual_cats[e.get('cat', 'unknown')] += 1
    
    flax_cats = defaultdict(int)
    for e in flax_all:
        flax_cats[e.get('cat', 'unknown')] += 1
    
    print(f"\nManual JIT Categories:")
    for cat, count in sorted(manual_cats.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {cat:30s} {count:>10,}")
    
    print(f"\nFlax Categories:")
    for cat, count in sorted(flax_cats.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {cat:30s} {count:>10,}")
    
    # Top operations by duration
    print("\n" + "=" * 100)
    print("TOP OPERATIONS BY DURATION")
    print("=" * 100)
    
    manual_ops = defaultdict(float)
    for e in manual_all:
        if e.get('ph') == 'X' and e.get('dur', 0) > 0:
            manual_ops[e.get('name', 'unknown')] += e.get('dur', 0) / 1000
    
    flax_ops = defaultdict(float)
    for e in flax_all:
        if e.get('ph') == 'X' and e.get('dur', 0) > 0:
            flax_ops[e.get('name', 'unknown')] += e.get('dur', 0) / 1000
    
    print(f"\nManual JIT (Top 30):")
    for i, (name, dur) in enumerate(sorted(manual_ops.items(), key=lambda x: x[1], reverse=True)[:30], 1):
        print(f"  {i:2d}. {name[:70]:70s} {dur:>10.2f} ms")
    
    print(f"\nFlax (Top 30):")
    for i, (name, dur) in enumerate(sorted(flax_ops.items(), key=lambda x: x[1], reverse=True)[:30], 1):
        print(f"  {i:2d}. {name[:70]:70s} {dur:>10.2f} ms")


if __name__ == "__main__":
    manual_jit_trace = Path("/mnt/carles/fme/trace_manual_jit/plugins/profile/2025_10_16_02_01_36/t1v-n-42336699-w-0.trace.json.gz")
    flax_trace = Path("/mnt/carles/fme/trace_flax_no_dropout/plugins/profile/2025_10_15_10_11_47/t1v-n-42336699-w-0.trace.json.gz")
    
    if not manual_jit_trace.exists():
        print(f"ERROR: Manual JIT trace not found at {manual_jit_trace}")
        print("Available traces:")
        for f in Path("/mnt/carles/fme/trace_manual_jit/plugins/profile").rglob("*.json.gz"):
            print(f"  {f}")
        exit(1)
    
    if not flax_trace.exists():
        print(f"ERROR: Flax trace not found at {flax_trace}")
        exit(1)
    
    print_trace_comparison(manual_jit_trace, flax_trace)
