# Roadmap

This document outlines the development roadmap for visdet, with a focus on performance improvements, modern integrations, and developer experience enhancements.

---

## SPDL Integration Plan

### Overview

SPDL (Scalable and Performant Data Loading) is Meta's thread-based data loading library that dramatically outperforms PyTorch's process-based `DataLoader`. Integrating SPDL into visdet will provide:

- **74% faster iteration** through datasets like ImageNet
- **38% less CPU usage** during training
- **50GB less memory** footprint compared to PyTorch DataLoader
- **Additional 33% speedup** with Free-Threaded Python 3.13t (no code changes needed)

### Why SPDL?

PyTorch's `DataLoader` uses multiprocessing, which has fundamental limitations:

| Aspect | PyTorch DataLoader | SPDL |
|--------|-------------------|------|
| Execution Model | Process-based (multiprocessing) | Thread-based |
| Memory | Copies batch tensors at least twice | Direct memory sharing |
| Scaling | Degrades beyond 8-16 workers | Scales linearly |
| Large Batches | Performance drops | Sustains/improves throughput |
| GIL Impact | N/A (separate processes) | Minimal (I/O bound), eliminated with Python 3.13t |

SPDL's explicit pipeline architecture also enables independent optimization of each stage:
- **Source**: Index/key generation
- **I/O**: File reading and decoding
- **Transform**: Preprocessing and augmentation
- **Batch**: Collation
- **Prefetch**: GPU transfer

### Current Architecture

visdet currently uses PyTorch's standard DataLoader in:

```
visdet/datasets/builder.py          # build_dataloader() function
visdet/engine/runner/runner.py      # Runner dataloader construction
visdet/runner.py                    # SimpleRunner dataloader config
```

Key components:
- `torch.utils.data.DataLoader` with multiprocessing workers
- Group samplers for aspect ratio grouping (`GroupSampler`, `DistributedGroupSampler`)
- Custom collate functions for detection data structures
- Worker initialization for reproducible seeding

### Integration Phases

#### Phase 1: Foundation (Target: Q1 2025)

**Goal**: Create SPDL adapter layer without breaking existing API

1. **Add SPDL as optional dependency**
   ```toml
   # pyproject.toml
   [project.optional-dependencies]
   spdl = ["spdl>=0.1.6"]
   ```

2. **Create `SPDLDataLoader` wrapper class**
   ```python
   # visdet/datasets/spdl_loader.py
   from spdl.dataloader import PipelineBuilder

   class SPDLDataLoader:
       """Drop-in replacement for PyTorch DataLoader using SPDL."""

       def __init__(
           self,
           dataset,
           batch_size: int,
           num_workers: int = 4,
           shuffle: bool = True,
           collate_fn = None,
           prefetch_factor: int = 2,
       ):
           self.dataset = dataset
           self.batch_size = batch_size
           self.num_workers = num_workers
           self.collate_fn = collate_fn or default_collate
           self.prefetch_factor = prefetch_factor
           self._build_pipeline(shuffle)

       def _build_pipeline(self, shuffle: bool):
           indices = list(range(len(self.dataset)))
           if shuffle:
               random.shuffle(indices)

           self.pipeline = (
               PipelineBuilder()
               .add_source(indices)
               .pipe(
                   self.dataset.__getitem__,
                   concurrency=self.num_workers,
                   output_order="input"  # Preserve order for reproducibility
               )
               .aggregate(self.batch_size)
               .pipe(self.collate_fn)
               .add_sink(self.prefetch_factor)
               .build(num_threads=self.num_workers)
           )

       def __iter__(self):
           return iter(self.pipeline)

       def __len__(self):
           return (len(self.dataset) + self.batch_size - 1) // self.batch_size
   ```

3. **Add configuration flag**
   ```python
   # In SimpleRunner
   runner = SimpleRunner(
       model='mask_rcnn_swin_s',
       dataset='coco_instance_segmentation',
       dataloader_backend='spdl',  # or 'pytorch' (default)
   )
   ```

#### Phase 2: Detection-Specific Optimizations (Target: Q2 2025)

**Goal**: Optimize SPDL pipeline for object detection workloads

1. **Aspect Ratio Grouping**
   - Implement group-aware batching in SPDL pipeline
   - Minimize padding waste by grouping similar aspect ratios

   ```python
   def aspect_ratio_group_source(dataset, batch_size):
       """Generate indices grouped by aspect ratio."""
       # Group images by aspect ratio buckets
       groups = defaultdict(list)
       for idx in range(len(dataset)):
           ar = dataset.get_aspect_ratio(idx)
           bucket = round(ar, 1)
           groups[bucket].append(idx)

       # Yield batches from same group
       for bucket, indices in groups.items():
           random.shuffle(indices)
           for i in range(0, len(indices), batch_size):
               yield indices[i:i + batch_size]
   ```

2. **Optimized Image Decoding**
   - Use SPDL's I/O utilities for efficient image loading
   - Avoid premature YUV→RGB conversion (~20MB savings per batch)
   - Leverage hardware decoders when available

3. **Transform Pipeline Optimization**
   - Profile transform bottlenecks
   - Parallelize independent transforms
   - Consider Kornia integration for GPU-accelerated augmentation

#### Phase 3: Distributed Training Support (Target: Q3 2025)

**Goal**: Full distributed training compatibility

1. **Distributed Sampling**
   - Implement SPDL-native distributed sampling
   - Ensure proper sharding across ranks
   - Support for `DistributedGroupSampler` equivalent

2. **Gradient Accumulation Compatibility**
   - Handle micro-batching for large effective batch sizes
   - Ensure correct behavior with mixed precision training

3. **Checkpoint/Resume Support**
   - Track pipeline state for training resume
   - Deterministic iteration with seed support

#### Phase 4: Advanced Features (Target: Q4 2025)

**Goal**: Leverage SPDL's full capabilities

1. **Python 3.13t Support**
   - Test with Free-Threaded Python
   - Document performance improvements
   - Add CI testing for nogil Python

2. **Dynamic Batching**
   - Variable batch sizes based on image complexity
   - Memory-aware batching to prevent OOM

3. **Streaming Datasets**
   - Support for datasets that don't fit in memory
   - Integration with cloud storage (S3, GCS)

4. **Performance Monitoring**
   - Built-in pipeline profiling
   - Bottleneck identification tools
   - Integration with training dashboards

### Migration Guide

For users migrating from PyTorch DataLoader:

```python
# Before (PyTorch DataLoader)
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoDataset', ...),
)

# After (SPDL)
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    backend='spdl',  # New flag
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoDataset', ...),
)
```

### Benchmarking Plan

We will benchmark SPDL integration against PyTorch DataLoader on:

| Metric | Measurement |
|--------|-------------|
| Throughput | Images/second during training |
| CPU Usage | Average CPU utilization |
| Memory | Peak and average RAM usage |
| GPU Utilization | Percentage of time GPU is active |
| Time to First Batch | Cold start latency |
| Scaling | Performance vs. num_workers |

Test configurations:
- Single GPU (RTX 3090, A100)
- Multi-GPU (4x, 8x A100)
- Various batch sizes (2, 4, 8, 16, 32)
- Dataset sizes (COCO, Objects365, custom)

### References

- SPDL GitHub Repository
- [SPDL Documentation](https://facebookresearch.github.io/spdl/main/)
- [Migration from PyTorch DataLoader](https://facebookresearch.github.io/spdl/main/migration/pytorch.html)
- [Meta AI Blog: Introducing SPDL](https://ai.meta.com/blog/spdl-faster-ai-model-training-with-thread-based-data-loading-reality-labs/)
- [SPDL Paper (arXiv:2504.20067)](https://arxiv.org/pdf/2504.20067)

---

## Other Planned Integrations

### Kornia (GPU-Accelerated Augmentations)

**Status**: Under evaluation

Differentiable augmentation pipelines that run on GPU, reducing CPU bottlenecks:
- Geometric transforms (rotation, affine, perspective)
- Color augmentations with gradient support
- Mosaic and mixup augmentations

### Triton (Custom Kernels)

**Status**: Research phase

Python-like GPU programming for custom operators:
- NMS acceleration
- RoI pooling/align optimization
- Attention mechanisms

### DALI (NVIDIA Data Loading)

**Status**: Evaluating alongside SPDL

GPU-accelerated preprocessing for NVIDIA hardware:
- Decode-on-GPU for reduced PCIe bandwidth
- Pipeline-parallel execution
- Best for high GPU:CPU ratio systems

---

## Timeline Summary

| Quarter | Milestone |
|---------|-----------|
| Q1 2025 | SPDL adapter layer, optional dependency |
| Q2 2025 | Detection-specific optimizations, aspect ratio grouping |
| Q3 2025 | Full distributed training support |
| Q4 2025 | Python 3.13t support, advanced features |
| 2026 | Kornia integration, Triton kernels |

---

## Contributing

We welcome contributions to any roadmap item! Please see the [Contributing Guide](development/contributing.md) for details on how to get involved.

Priority areas for contribution:
- SPDL integration testing and benchmarking
- Distributed training validation
- Documentation and examples
