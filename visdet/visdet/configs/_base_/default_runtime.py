# ruff: noqa
# fmt: off
# isort: skip
# type: ignore

default_scope = None

default_hooks = {
    "timer": {"type": "IterTimerHook"},
    "logger": {"type": "LoggerHook", "interval": 50},
    "param_scheduler": {"type": "ParamSchedulerHook"},
    "checkpoint": {"type": "CheckpointHook", "interval": 1},
    "sampler_seed": {"type": "DistSamplerSeedHook"},
    "visualization": {"type": "DetVisualizationHook"},
}

env_cfg = {
    "cudnn_benchmark": False,
    "mp_cfg": {"mp_start_method": "fork", "opencv_num_threads": 0},
    "dist_cfg": {"backend": "nccl"},
}

vis_backends = [{"type": "LocalVisBackend"}]
visualizer = {"type": "DetLocalVisualizer", "vis_backends": vis_backends, "name": "visualizer"}
log_processor = {"type": "LogProcessor", "window_size": 50, "by_epoch": True}

log_level = "INFO"
load_from = None
resume = False
