# Optimizers

visdet uses a registry-based system for optimizers. In configs, you typically specify an optimizer by its class name:

```python
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05)
```

In MMEngine-style configs you can also place this under `optim_wrapper.optimizer`.

## Built-in (PyTorch)

visdet automatically registers optimizers from `torch.optim`. The exact set depends on your PyTorch version, but the following are commonly available.

| `type` | Developed | Summary |
| --- | ---: | --- |
| `SGD` | 1951[^sgd] | Classic stochastic gradient descent; supports momentum and Nesterov momentum via configuration. |
| `ASGD` | 1992[^asgd] | SGD variant that maintains an average of parameters, which can improve convergence on some problems. |
| `Rprop` | 1993[^rprop] | Uses only the sign of gradients and adapts per-parameter step sizes; mainly used in smaller/older setups. |
| `LBFGS` | 1989[^lbfgs] | Limited-memory quasi-Newton optimizer; most useful for small problems or near-convex objectives. |
| `Adagrad` | 2011[^adagrad] | Per-parameter adaptive learning rates; often helpful for sparse features but can decay learning rates aggressively. |
| `Adadelta` | 2012[^adadelta] | Adagrad-like method using running averages to reduce sensitivity to the initial learning rate. |
| `RMSprop` | 2012[^rmsprop] | Adaptive learning rates using an EMA of squared gradients; commonly effective for non-stationary objectives. |
| `Adam` | 2014[^adam] | Adaptive moment estimation using EMAs of gradients and squared gradients; a common default. |
| `Adamax` | 2014[^adam] | Adam variant based on the infinity norm of past gradients; can be more stable in some regimes. |
| `SparseAdam` | 2014[^adam] | Adam variant intended for sparse gradients (e.g., embeddings); not suitable for dense parameters. |
| `NAdam` | 2016[^nadam] | Adam with Nesterov-style momentum; can sometimes improve early training dynamics. |
| `AdamW` | 2017[^adamw] | Adam with decoupled weight decay; often yields better generalization than L2-style weight decay in Adam. |
| `TorchAdafactor` | 2018[^adafactor] | PyTorch’s Adafactor implementation (registered as `TorchAdafactor` when available). |
| `RAdam` | 2019[^radam] | Rectified Adam; adds variance rectification to reduce warmup sensitivity. |
| `Muon` | 2024[^muon] | Optimizer designed for 2D weight matrices using Newton–Schulz orthogonalization; typically applied selectively. |

## Optional integrations

These optimizers are registered only if their dependencies are installed.

### Hugging Face Transformers

| `type` | Developed | Summary |
| --- | ---: | --- |
| `Adafactor` | 2018[^adafactor] | Transformers’ Adafactor implementation (registered as `Adafactor`). |

### bitsandbytes (8-bit optimizers)

| `type` | Developed | Summary |
| --- | ---: | --- |
| *(varies)* | 2021[^bnb8bit] | visdet registers all optimizer classes exposed by `bitsandbytes.optim`; name collisions are prefixed with `bnb_`. |

### D-Adaptation

| `type` | Developed | Summary |
| --- | ---: | --- |
| `DAdaptAdam` | 2023[^dadapt] | Learning-rate-free variant of Adam that adapts the step size automatically. |
| `DAdaptAdaGrad` | 2023[^dadapt] | Learning-rate-free variant of Adagrad. |
| `DAdaptSGD` | 2023[^dadapt] | Learning-rate-free variant of SGD. |

### Lion

| `type` | Developed | Summary |
| --- | ---: | --- |
| `Lion` | 2023[^lion] | Sign-based optimizer discovered via symbolic search; often competitive with AdamW at lower compute overhead. |

### Sophia

| `type` | Developed | Summary |
| --- | ---: | --- |
| *(varies)* | 2023[^sophia] | visdet registers any Sophia optimizer classes found in the installed `Sophia` package (commonly `SophiaG`). |

### DeepSpeed

| `type` | Developed | Summary |
| --- | ---: | --- |
| `DeepSpeedCPUAdam` | 2014[^adam] | DeepSpeed CPU-optimized Adam implementation for large-scale training. |
| `FusedAdam` | 2014[^adam] | Fused GPU Adam implementation to reduce optimizer overhead. |
| `FusedLamb` | 2019[^lamb] | Fused LAMB implementation for very large-batch training. |
| `OnebitAdam` | 2014[^adam] | Communication-efficient Adam variant for distributed training. |
| `OnebitLamb` | 2019[^lamb] | Communication-efficient LAMB variant for distributed training. |
| `ZeroOneAdam` | 2014[^adam] | DeepSpeed optimizer variant used with ZeRO-style training setups. |

## Listing available optimizers

Because the available set can change depending on installed optional dependencies, you can list what your environment has registered:

```python
from visdet.engine.optim import OPTIMIZERS

print(sorted(OPTIMIZERS.module_dict.keys()))
```

!!! note
    You may see `Optimizer` in the registry output; it is the base PyTorch optimizer class and is not intended to be used as a config `type`.

[^sgd]: Robbins & Monro (1951), *A Stochastic Approximation Method*. https://doi.org/10.1214/aoms/1177729586
[^asgd]: Polyak & Juditsky (1992), *Acceleration of Stochastic Approximation by Averaging*. https://doi.org/10.1137/0330046
[^rprop]: Riedmiller & Braun (1993), *A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm*.
[^lbfgs]: Liu & Nocedal (1989), *On the Limited Memory BFGS Method for Large Scale Optimization*. https://doi.org/10.1007/BF01589116
[^adagrad]: Duchi, Hazan & Singer (2011), *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*. https://jmlr.org/papers/v12/duchi11a.html
[^adadelta]: Zeiler (2012), *ADADELTA: An Adaptive Learning Rate Method*. https://arxiv.org/abs/1212.5701
[^rmsprop]: Tieleman & Hinton (2012), *RMSProp* (lecture note; widely referenced).
[^adam]: Kingma & Ba (2014), *Adam: A Method for Stochastic Optimization*. https://arxiv.org/abs/1412.6980
[^nadam]: Dozat (2016), *Incorporating Nesterov Momentum into Adam*. https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
[^adamw]: Loshchilov & Hutter (2017), *Decoupled Weight Decay Regularization*. https://arxiv.org/abs/1711.05101
[^adafactor]: Shazeer & Stern (2018), *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*. https://arxiv.org/abs/1804.04235
[^radam]: Liu et al. (2019), *On the Variance of the Adaptive Learning Rate and Beyond*. https://arxiv.org/abs/1908.03265
[^lamb]: You et al. (2019), *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes*. https://arxiv.org/abs/1904.00962
[^bnb8bit]: Dettmers et al. (2021), *8-bit Optimizers via Block-wise Quantization*. https://arxiv.org/abs/2110.02861
[^dadapt]: Samuel et al. (2023), *Learning-Rate-Free Learning by D-Adaptation*. https://arxiv.org/abs/2301.07733
[^lion]: Chen et al. (2023), *Symbolic Discovery of Optimization Algorithms*. https://arxiv.org/abs/2302.06675
[^sophia]: Liu et al. (2023), *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*. https://arxiv.org/abs/2305.14342
[^muon]: Jordan (2024), *Muon: An optimizer for hidden layers in neural networks*. https://kellerjordan.github.io/posts/muon/
