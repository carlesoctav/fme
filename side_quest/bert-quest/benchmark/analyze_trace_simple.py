#!/usr/bin/env python3
"""
Simple trace analyzer that uses protobuf directly.
"""

import sys
from pathlib import Path
from collections import defaultdict

try:
    from tensorflow.core.profiler.protobuf import xplane_pb2
except ImportError:
    print("ERROR: tensorflow not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflow"])
    from tensorflow.core.profiler.protobuf import xplane_pb2


def analyze_trace(xplane_path: str):
    """Analyze XPlane protobuf trace."""
    print(f"\nAnalyzing: {xplane_path}")
    print("=" * 80)
    
    with open(xplane_path, "rb") as f:
        xspace = xplane_pb2.XSpace()
        xspace.ParseFromString(f.read())
    
    print(f"Found {len(xspace.planes)} planes in trace")
    
    # Collect all events
    all_events = []
    
    for plane_idx, plane in enumerate(xspace.planes):
        # Build lookup tables
        stat_metadata = {m.id: m.name for m in plane.stat_metadata_list}
        event_metadata = {m.id: m.name for m in plane.event_metadata_list}
        
        print(f"\nPlane {plane_idx}: {plane.name if hasattr(plane, 'name') else 'unnamed'}")
        print(f"  Events: {sum(len(line.events) for line in plane.lines)}")
        print(f"  Lines: {len(plane.lines)}")
        
        for line in plane.lines:
            for event in line.events:
                event_name = event_metadata.get(event.metadata_id, f"Event_{event.metadata_id}")
                duration_ms = event.duration_ps / 1e9
                
                all_events.append({
                    "name": event_name,
                    "duration_ms": duration_ms,
                    "plane": plane_idx,
                })
    
    # Sort by duration
    all_events.sort(key=lambda x: x["duration_ms"], reverse=True)
    
    # Show top operations
    print("\n" + "=" * 80)
    print("TOP 50 OPERATIONS BY DURATION")
    print("=" * 80)
    print(f"{'Operation':<70} {'Duration (ms)':>15}")
    print("-" * 86)
    
    for i, event in enumerate(all_events[:50]):
        if event["duration_ms"] < 0.01:  # Skip very short ops
            continue
        name = event["name"]
        if len(name) > 70:
            name = name[:67] + "..."
        print(f"{name:<70} {event['duration_ms']:>15.2f}")
    
    # Aggregate by keywords
    print("\n" + "=" * 80)
    print("AGGREGATED BY KEYWORD")
    print("=" * 80)
    
    keywords = [
        "tree", "pytree", "flatten", "unflatten", "leaves",
        "filter", "shard", "partition", "transform",
        "transpose", "copy", "memcpy", "collective",
        "jit", "compile", "xla",
    ]
    
    aggregates = defaultdict(lambda: {"total_ms": 0.0, "count": 0})
    
    for event in all_events:
        name_lower = event["name"].lower()
        for keyword in keywords:
            if keyword in name_lower:
                aggregates[keyword]["total_ms"] += event["duration_ms"]
                aggregates[keyword]["count"] += 1
    
    # Sort by total time
    sorted_agg = sorted(aggregates.items(), key=lambda x: x[1]["total_ms"], reverse=True)
    
    print(f"{'Keyword':<30} {'Total (ms)':>15} {'Count':>10} {'Avg (ms)':>15}")
    print("-" * 80)
    for keyword, data in sorted_agg:
        if data["total_ms"] >= 0.1:
            avg = data["total_ms"] / data["count"] if data["count"] > 0 else 0
            print(f"{keyword:<30} {data['total_ms']:>15.2f} {data['count']:>10} {avg:>15.4f}")


def main():
    trace_path = "trace_make_module_opt_batched/plugins/profile/2025_10_16_02_57_18/t1v-n-42336699-w-0.xplane.pb"
    
    if not Path(trace_path).exists():
        print(f"Trace file not found: {trace_path}")
        # Try to find any xplane file
        trace_dir = Path("trace_make_module_opt_batched")
        xplane_files = list(trace_dir.rglob("*.xplane.pb"))
        if xplane_files:
            trace_path = str(xplane_files[0])
            print(f"Using: {trace_path}")
        else:
            print("No trace files found!")
            return
    
    analyze_trace(trace_path)


if __name__ == "__main__":
    main()
