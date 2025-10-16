#!/usr/bin/env python3
"""
Analyze JAX profiler trace from make_module_opt to understand timing breakdown.
"""

import sys
from pathlib import Path

# Add tensorboard path if needed
try:
    from tensorboard.plugins.profile.trace_events_pb2 import Trace
except ImportError:
    print("Installing tensorboard...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorboard"])
    from tensorboard.plugins.profile.trace_events_pb2 import Trace

from tensorflow.core.profiler.protobuf import xplane_pb2


def analyze_xplane(xplane_path: str):
    """Analyze XPlane trace to extract timing information."""
    print(f"\nAnalyzing trace: {xplane_path}")
    print("=" * 80)
    
    with open(xplane_path, "rb") as f:
        xspace = xplane_pb2.XSpace()
        xspace.ParseFromString(f.read())
    
    # Extract timing data from XPlane
    all_events = []
    
    for plane in xspace.planes:
        # Get string table for this plane
        stat_metadata = {m.id: m.name for m in plane.stat_metadata_list}
        event_metadata = {m.id: m.name for m in plane.event_metadata_list}
        
        for line in plane.lines:
            for event in line.events:
                # Get event name
                event_name = event_metadata.get(event.metadata_id, f"UnknownEvent_{event.metadata_id}")
                
                # Get duration (in picoseconds, convert to ms)
                duration_ps = event.duration_ps
                duration_ms = duration_ps / 1e9
                
                # Get stats (additional metadata)
                stats = {}
                for stat in event.stats:
                    stat_name = stat_metadata.get(stat.metadata_id, f"UnknownStat_{stat.metadata_id}")
                    if stat.HasField("int64_value"):
                        stats[stat_name] = stat.int64_value
                    elif stat.HasField("uint64_value"):
                        stats[stat_name] = stat.uint64_value
                    elif stat.HasField("double_value"):
                        stats[stat_name] = stat.double_value
                    elif stat.HasField("str_value"):
                        stats[stat_name] = stat.str_value
                
                all_events.append({
                    "name": event_name,
                    "duration_ms": duration_ms,
                    "timestamp_ps": event.offset_ps,
                    "stats": stats,
                })
    
    # Sort by duration (longest first)
    all_events.sort(key=lambda x: x["duration_ms"], reverse=True)
    
    # Print top operations
    print("\nTop operations by duration:")
    print("-" * 80)
    print(f"{'Operation':<60} {'Duration (ms)':>15}")
    print("-" * 80)
    
    # Filter for interesting operations
    shown = 0
    for event in all_events[:100]:  # Top 100
        name = event["name"]
        duration = event["duration_ms"]
        
        # Skip very short operations
        if duration < 0.1:
            continue
        
        # Show interesting operations
        if any(keyword in name.lower() for keyword in [
            "tree", "filter", "partition", "shard", "apply", "transform",
            "module", "opt", "build", "jit", "compile", "transpose", "copy"
        ]):
            print(f"{name:<60} {duration:>15.2f}")
            shown += 1
            if shown >= 30:
                break
    
    # Aggregate by operation type
    print("\n" + "=" * 80)
    print("Aggregated timing by operation type:")
    print("-" * 80)
    
    aggregates = {}
    for event in all_events:
        name = event["name"]
        duration = event["duration_ms"]
        
        # Categorize operations
        category = "other"
        if "tree_at" in name.lower() or "tree-at" in name.lower():
            category = "tree_at"
        elif "tree_map" in name.lower() or "tree-map" in name.lower():
            category = "tree_map"
        elif "tree_flatten" in name.lower() or "tree-flatten" in name.lower():
            category = "tree_flatten"
        elif "filter_shard" in name.lower() or "filter-shard" in name.lower():
            category = "filter_shard"
        elif "partition" in name.lower():
            category = "partition"
        elif "apply_transforms" in name.lower():
            category = "apply_transforms"
        elif "transpose" in name.lower() or "copy" in name.lower():
            category = "data_movement"
        elif "compile" in name.lower() or "jit" in name.lower():
            category = "compilation"
        
        if category not in aggregates:
            aggregates[category] = {"total_ms": 0, "count": 0}
        aggregates[category]["total_ms"] += duration
        aggregates[category]["count"] += 1
    
    # Sort by total time
    sorted_agg = sorted(aggregates.items(), key=lambda x: x[1]["total_ms"], reverse=True)
    
    print(f"{'Category':<30} {'Total (ms)':>15} {'Count':>10} {'Avg (ms)':>15}")
    print("-" * 80)
    for category, data in sorted_agg[:20]:
        if data["total_ms"] >= 0.1:  # Only show significant categories
            avg = data["total_ms"] / data["count"] if data["count"] > 0 else 0
            print(f"{category:<30} {data['total_ms']:>15.2f} {data['count']:>10} {avg:>15.2f}")
    
    # Look for specific functions
    print("\n" + "=" * 80)
    print("Searching for key functions:")
    print("-" * 80)
    
    key_functions = [
        "apply_transforms", "tree_at", "filter_shard", "get_partition_spec",
        "Optimizer.create", "_build", "filter_jit"
    ]
    
    for func_name in key_functions:
        matching = [e for e in all_events if func_name.lower() in e["name"].lower()]
        if matching:
            total = sum(e["duration_ms"] for e in matching)
            count = len(matching)
            avg = total / count if count > 0 else 0
            print(f"{func_name:<30} {total:>10.2f}ms total, {count:>5} calls, {avg:>10.2f}ms avg")
        else:
            print(f"{func_name:<30} {'NOT FOUND':>10}")


def main():
    trace_dir = Path("trace_make_module_opt_batched")
    
    # Find xplane file
    xplane_files = list(trace_dir.rglob("*.xplane.pb"))
    
    if not xplane_files:
        print(f"No xplane.pb files found in {trace_dir}")
        return
    
    print(f"Found {len(xplane_files)} trace file(s)")
    
    # Analyze the first one (usually the most recent)
    analyze_xplane(str(xplane_files[0]))


if __name__ == "__main__":
    main()
