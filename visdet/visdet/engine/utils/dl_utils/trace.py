import torch


# This is not worth a whole file
def is_jit_tracing() -> bool:
    return torch.jit.is_tracing()
