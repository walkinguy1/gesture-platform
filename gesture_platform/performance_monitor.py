"""
Performance Monitor Module
Provides timing and performance tracking for the gesture platform.

Useful for:
- Measuring inference latency
- Tracking FPS over time
- Identifying performance bottlenecks
- Profiling component performance
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timing measurement."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_time(self) -> float:
        """Average time in milliseconds."""
        if self.count == 0:
            return 0.0
        return (self.total_time / self.count) * 1000

    @property
    def recent_avg(self) -> float:
        """Average of recent times in milliseconds."""
        if not self.recent_times:
            return 0.0
        return (sum(self.recent_times) / len(self.recent_times)) * 1000

    def add_sample(self, duration: float) -> None:
        """Add a timing sample."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking timing metrics.

    Example:
        monitor = PerformanceMonitor()

        with monitor.time("inference"):
            result = model.predict(features)

        stats = monitor.get_stats("inference")
        print(f"Average inference time: {stats.avg_time:.2f}ms")
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the performance monitor.

        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self._stats: Dict[str, TimingStats] = {}
        self._current_timings: Dict[str, float] = {}

    @contextmanager
    def time(self, name: str):
        """
        Context manager for timing a block of code.

        Args:
            name: Name for the timing measurement

        Example:
            with monitor.time("feature_extraction"):
                features = extractor.extract(landmarks)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        self._current_timings[name] = start

        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._record_timing(name, duration)
            if name in self._current_timings:
                del self._current_timings[name]

    def _record_timing(self, name: str, duration: float) -> None:
        """Record a timing measurement."""
        if name not in self._stats:
            self._stats[name] = TimingStats(name=name)
        self._stats[name].add_sample(duration)

        # Log slow operations
        if duration > 0.1:  # 100ms threshold
            logger.warning(
                "Slow operation detected: %s took %.2fms",
                name,
                duration * 1000
            )

    def get_stats(self, name: str) -> Optional[TimingStats]:
        """Get statistics for a named operation."""
        return self._stats.get(name)

    def get_all_stats(self) -> Dict[str, TimingStats]:
        """Get all timing statistics."""
        return self._stats.copy()

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset statistics.

        Args:
            name: Specific name to reset, or None to reset all
        """
        if name:
            if name in self._stats:
                del self._stats[name]
        else:
            self._stats.clear()

    def print_summary(self) -> None:
        """Print a summary of all timing statistics."""
        if not self._stats:
            logger.info("No performance statistics available.")
            return

        logger.info("=" * 60)
        logger.info("Performance Summary")
        logger.info("=" * 60)

        for name, stats in sorted(self._stats.items(), key=lambda x: x[1].avg_time, reverse=True):
            logger.info(
                "%-30s: %.2fms avg (min: %.2fms, max: %.2fms, count: %d)",
                name,
                stats.avg_time,
                stats.min_time * 1000 if stats.min_time != float('inf') else 0,
                stats.max_time * 1000,
                stats.count
            )

        logger.info("=" * 60)

    def get_report(self) -> Dict[str, Dict]:
        """
        Get a performance report as a dictionary.

        Returns:
            Dictionary with timing statistics for all operations
        """
        report = {}
        for name, stats in self._stats.items():
            report[name] = {
                "avg_ms": stats.avg_time,
                "recent_avg_ms": stats.recent_avg,
                "min_ms": stats.min_time * 1000 if stats.min_time != float('inf') else 0,
                "max_ms": stats.max_time * 1000,
                "count": stats.count
            }
        return report


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor


def time_operation(name: str):
    """
    Decorator for timing function execution.

    Args:
        name: Name for the timing measurement

    Example:
        @time_operation("inference")
        def predict(features):
            return model.predict(features)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with _global_monitor.time(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
