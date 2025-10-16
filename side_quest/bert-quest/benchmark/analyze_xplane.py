#!/usr/bin/env python3
"""
Analyze XLA profiler xplane.pb files to extract kernel execution times.
Uses TensorBoard's profiler plugin to parse the binary format.
"""

import sys
from pathlib import Path

try:
    from tensorflow.core.profiler import profiler_analysis_pb2
    from tensorflow.python.profiler import trace_events_pb2
    import tensorflow as tf
except ImportError:
    print("TensorFlow not available, trying alternative approach...")
    sys.exit(1)

def analyze_xplane(xplane_path):
    """Parse xplane.pb file and extract timing information."""
    
    print(f"\nAnalyzing: {xplane_path}")
    print("=" * 80)
    
    # Try to load and analyze using TensorBoard profiler
    try:
        from tensorboard.plugins.profile import trace_events_json
        
        # Read the xplane file
        with open(xplane_path, 'rb') as f:
            data = f.read()
        
        print(f"File size: {len(data) / 1024 / 1024:.2f} MB")
        print(f"First 100 bytes: {data[:100]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    manual_jit = Path("/mnt/carles/fme/trace_manual_jit/plugins/profile/2025_10_15_05_52_01/t1v-n-42336699-w-0.xplane.pb")
    flax = Path("/mnt/carles/fme/trace_flax_no_dropout/plugins/profile/2025_10_15_10_11_47/t1v-n-42336699-w-0.xplane.pb")
    
    analyze_xplane(manual_jit)
    analyze_xplane(flax)
