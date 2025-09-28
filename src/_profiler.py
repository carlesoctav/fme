"""Backward compatibility shim for the profiler callback."""

from __future__ import annotations

from .callbacks.jax_profiler import JaxProfiler


__all__ = ["JaxProfiler"]
