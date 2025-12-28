# Blue Sky Ideas

This document captures experimental directions and research ideas for visdet. These are speculative explorations—some may prove valuable, others may not pan out. The goal is to document interesting possibilities for the community to explore.

---

## Pluggable Attention Architectures

Modern vision transformers use various attention mechanisms with different tradeoffs. Making these easily swappable would enable rapid experimentation.

### Candidates

| Architecture | Description | Potential Benefit |
|--------------|-------------|-------------------|
| **MLA** (Multi-head Latent Attention) | DeepSeek's compressed KV cache approach | Reduced memory, faster inference |
| **SWA** (Sliding Window Attention) | Local attention with fixed window size | Linear complexity for long sequences |
| **NSA** (Native Sparse Attention) | Hardware-aware sparse attention patterns | Better GPU utilization |
| **KDA** (Kernel Density Attention) | Kernel-based attention approximation | Theoretical efficiency gains |

### Design Goals

- Single config change to swap attention mechanism
- Unified interface across all attention types
- Automatic fallback to standard attention if unsupported
- Benchmark suite comparing mechanisms on detection tasks

```python
# Hypothetical config
backbone = dict(
    type='SwinTransformer',
    attention_type='mla',  # or 'swa', 'nsa', 'standard'
    attention_cfg=dict(
        window_size=7,
        compression_ratio=4,
    ),
)
```

---

## Multi-Precision Training

Beyond standard FP16/BF16 mixed precision, explore more granular precision control:

- **FP8 training**: Emerging support in modern GPUs (H100+)
- **Per-layer precision**: Different precisions for backbone vs. head
- **Dynamic precision**: Adjust based on gradient statistics
- **INT8 forward / FP16 backward**: Aggressive memory savings

### Research Questions

- How does precision affect detection metrics (mAP, small object recall)?
- Can we use lower precision for early backbone layers?
- What's the memory/accuracy tradeoff for FP8 in detection heads?

---

## Multi-Optimizer Configurations

Recent research suggests different model components may benefit from different optimizers.

### Potential Setups

| Component | Optimizer | Rationale |
|-----------|-----------|-----------|
| Backbone (pretrained) | SGD with low LR | Stability, preserve pretrained features |
| Neck | AdamW | Fast adaptation |
| Head | LAMB / Shampoo | Fresh weights, can be aggressive |

### New Optimizers to Evaluate

- **Shampoo**: Second-order optimizer with structured preconditioning
- **LAMB**: Layer-wise adaptive scaling for large batch training
- **Lion**: Google's memory-efficient optimizer
- **Sophia**: Hessian-based with clipping
- **Schedule-Free**: Eliminates LR schedule tuning

### Implementation Sketch

```python
optimizer = dict(
    type='MultiOptimizer',
    constructors=[
        dict(
            type='SGD',
            lr=0.001,
            paramwise_cfg=dict(
                custom_keys={'backbone': dict(apply=True)}
            )
        ),
        dict(
            type='AdamW',
            lr=0.0001,
            paramwise_cfg=dict(
                custom_keys={'neck': dict(apply=True)}
            )
        ),
        dict(
            type='Lion',
            lr=0.00003,
            paramwise_cfg=dict(
                custom_keys={'head': dict(apply=True)}
            )
        ),
    ]
)
```

---

## Alternative Training Frameworks

While visdet uses PyTorch, integrating with specialized training frameworks could unlock performance:

### Evaluated (Issues Found)

| Framework | Status | Notes |
|-----------|--------|-------|
| **NeMo** | Explored | Primarily NLP-focused, detection support limited |
| **Megatron** | Explored | Great for LLMs, overkill for vision |
| **TorchTitan** | Explored | Promising but early stage |

### Worth Watching

- **MaxText/Pax**: JAX-based, interesting parallelism strategies
- **Composer**: MosaicML's training library, good abstractions
- **Levanter**: JAX training with named axes

---

## Speculative Ideas

### Mixture of Experts for Detection

- Route different object scales to specialized experts
- Potential for better small vs. large object handling
- Challenge: maintaining spatial coherence

### Neural Architecture Search for Necks

- FPN alternatives are often hand-designed
- NAS could discover better feature fusion patterns
- Could target specific hardware (mobile, edge, server)

### Continuous Learning / Streaming Training

- Update model on new data without full retraining
- Important for production deployment
- Challenge: catastrophic forgetting for rare classes

### Self-Supervised Pretraining for Detection

- Move beyond ImageNet classification pretraining
- Dense prediction pretraining (MAE, BEiT variants)
- Detection-aware pretraining objectives

---

## Contributing Ideas

Have a blue-sky idea? Open a discussion on GitHub! We're especially interested in:

1. Novel attention mechanisms with detection-specific benefits
2. Training efficiency improvements
3. Ideas from other domains (NLP, audio) that could transfer

No idea is too speculative for this document—the goal is to capture possibilities, not commitments.
