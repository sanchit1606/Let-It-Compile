"""
Metric definitions and normalization for profiling data.

Defines standard metric formats, units, and normalization functions
used across the profiling and RL modules.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Metric:
    """Definition of a performance metric."""
    name: str
    unit: str
    description: str
    min_val: float = 0.0
    max_val: float = 100.0


# Standard metrics for GPU profiling
METRICS = {
    "occupancy": Metric(
        name="Occupancy",
        unit="%",
        description="Percentage of maximum warps active",
        min_val=0.0,
        max_val=100.0
    ),
    "l2_hit_rate": Metric(
        name="L2 Hit Rate",
        unit="%",
        description="Percentage of L2 cache hits",
        min_val=0.0,
        max_val=100.0
    ),
    "dram_bandwidth": Metric(
        name="DRAM Bandwidth",
        unit="% of peak",
        description="Percentage of peak DRAM bandwidth utilized",
        min_val=0.0,
        max_val=100.0
    ),
    "warp_efficiency": Metric(
        name="Warp Efficiency",
        unit="%",
        description="Percentage of warp slots executing valid instructions",
        min_val=0.0,
        max_val=100.0
    ),
    "register_usage": Metric(
        name="Registers per Thread",
        unit="regs",
        description="Number of 32-bit registers used per thread",
        min_val=0.0,
        max_val=256.0
    ),
}


def normalize_metric(value: float, metric_name: str) -> float:
    """
    Normalize a metric value to [0, 1] range.

    Args:
        value: Metric value in original units
        metric_name: Name of the metric from METRICS dict

    Returns:
        Normalized value in [0, 1]
    """
    if metric_name not in METRICS:
        return float(value)

    metric = METRICS[metric_name]
    range_val = metric.max_val - metric.min_val

    if range_val == 0:
        return 0.0

    normalized = (value - metric.min_val) / range_val
    return max(0.0, min(1.0, normalized))


def denormalize_metric(normalized: float, metric_name: str) -> float:
    """
    Denormalize a metric from [0, 1] to original units.

    Args:
        normalized: Value in [0, 1]
        metric_name: Name of the metric

    Returns:
        Value in original units
    """
    if metric_name not in METRICS:
        return float(normalized)

    metric = METRICS[metric_name]
    range_val = metric.max_val - metric.min_val

    return metric.min_val + normalized * range_val
