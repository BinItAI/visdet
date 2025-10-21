# visengine

This is a simplified version of MMEngine, focused on supporting training and inference for Swin Mask R-CNN.

## Key Principles

1. **Simplified Runner**: Keep only the essential training loop functionality
2. **Basic Hooks**: Only maintain hooks actually used by Swin Mask R-CNN training
3. **No Complex Strategies**: Remove distributed training strategies we don't need

## What to Keep

- Basic Runner class for training/validation loops
- Essential hooks:
  - CheckpointHook
  - LoggerHook
  - OptimizerHook
  - IterTimerHook
- Config system (simplified)
- Registry system
- Basic data structures
- File I/O utilities

## What to Remove

- Complex distributed strategies (keep only single GPU and basic DDP)
- Unused hooks and runners
- Advanced features not needed for Swin Mask R-CNN
- Profiling and debugging tools we don't use

## Dependencies

Should only depend on:
- PyTorch
- Basic Python packages (numpy, etc.)
- viscv for image operations

---

*For machine learning guidelines, see the machine_learning/AGENTS.md file.*
*For general repository guidelines, see the root AGENTS.md file.*
