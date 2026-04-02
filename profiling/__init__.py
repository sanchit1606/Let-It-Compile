"""
GPU profiling and instrumentation modules.

Modules:
- cuda_timer.py: Precise kernel timing via CUDA events
- cupti_collector.py: CUPTI-derived metrics via Nsight Compute (ncu)
- ncu_utils.py: Nsight Compute (ncu) smoke tests + permission diagnostics
- nvml_monitor.py: Real-time GPU monitoring via pynvml
- metrics.py: Metric definitions and normalization
"""
