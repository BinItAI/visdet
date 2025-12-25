# Optimizers

visdet uses a registry-based system for optimizers. In configs, you typically specify an optimizer by its class name:

```python
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05)
```

In MMEngine-style configs you can also place this under `optim_wrapper.optimizer`.

## Core Optimizers

### SGD (Stochastic Gradient Descent)

The simplest optimization algorithm that updates parameters in the direction of the negative gradient.[^sgd]

```
params = params - learning_rate * gradients
```

With momentum, SGD maintains a velocity term that accumulates past gradients:

```
velocity = momentum * velocity - learning_rate * gradients
params = params + velocity
```

**Key Hyperparameters:**

- Learning rate (typically 0.01-0.1)
- Momentum (typically 0.9 when used)
- Weight decay (typically 1e-4 to 1e-5)

**Characteristics:**

- Simple implementation
- Often requires manual learning rate scheduling
- Can struggle with saddle points and ravines in the loss landscape
- With momentum, can achieve good performance but requires careful tuning

### Adam (Adaptive Moment Estimation)

Adam adapts learning rates for each parameter based on first and second moments of gradients.[^adam]

```
m = beta1 * m + (1 - beta1) * gradients           # First moment estimate
v = beta2 * v + (1 - beta2) * gradients^2         # Second moment estimate
m_hat = m / (1 - beta1^t)                         # Bias correction
v_hat = v / (1 - beta2^t)                         # Bias correction
params = params - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

**Key Hyperparameters:**

- Learning rate (typically 0.001)
- Beta1: Exponential decay rate for first moment (typically 0.9)
- Beta2: Exponential decay rate for second moment (typically 0.999)
- Epsilon: Small value for numerical stability (typically 1e-8)

**Characteristics:**

- Adaptive learning rates for each parameter
- Works well on a wide range of problems
- Less sensitive to hyperparameter choices than SGD
- Can converge quickly but sometimes generalization is not as good as SGD with momentum

### AdamW

A variant of Adam that decouples weight decay from the gradient update, implementing weight decay correctly.[^adamw]

```
m = beta1 * m + (1 - beta1) * gradients
v = beta2 * v + (1 - beta2) * gradients^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
params = params - learning_rate * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * params)
```

**Key Hyperparameters:**

- Same as Adam
- Weight decay is applied separately from the gradient update (typically 0.01-0.1)

**Characteristics:**

- Better generalization than standard Adam
- Properly decouples weight decay from adaptive learning rate
- Combines the benefits of Adam's adaptivity with proper regularization
- Currently considered state-of-the-art for many deep learning tasks

### Adagrad/AdaDelta/RMSprop

These are adaptive gradient methods that adjust learning rates based on historical gradient information.[^adagrad][^adadelta][^rmsprop]

**RMSprop update:**
```
v = decay * v + (1 - decay) * gradients^2
params = params - learning_rate * gradients / (sqrt(v) + epsilon)
```

**Key Characteristics:**

- Adagrad accumulates squared gradients, which can lead to premature stopping for deep networks
- AdaDelta and RMSprop address this by using a moving average of squared gradients
- Generally superseded by Adam in most applications

---

## Our Standard: AdamW

We exclusively use AdamW across all our training tasks for the following reasons:

**Better Generalization:** AdamW implements weight decay correctly, leading to better model generalization.

**Convergence Speed:** Maintains Adam's fast convergence properties while improving final model quality.

**Robustness:** Works well across a variety of network architectures and tasks without extensive tuning.

**State-of-the-Art:** Consistently delivers strong performance on modern deep learning tasks.

### Typical Configuration

Our standard AdamW configuration:

```python
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)
```

We pair this with a learning rate scheduler (typically cosine with warmup) for optimal performance across our training regimes.

---

## Modern Optimizers (2023-2024)

In addition to AdamW, we support several modern optimizers that offer different trade-offs. These are available via the optimizer registry and can be configured through the training config.

### Lion (2023)

Lion (EvoLved Sign Momentum) is an optimizer discovered through program search by Google Research.[^lion] It uses sign-based updates, resulting in uniform magnitude updates and significant memory savings.

```
update = beta1 * m + (1 - beta1) * gradients
m = beta2 * m + (1 - beta2) * gradients
params = params - learning_rate * (sign(update) + weight_decay * params)
```

**Key Characteristics:**

- Uses sign of momentum for updates (uniform magnitude)
- Requires 3-10x smaller learning rate than AdamW (e.g., 1e-4 vs 3e-4)
- Requires 3-10x larger weight decay than AdamW
- Best for batch sizes >= 64
- 50% less memory than Adam (only one momentum buffer)

```python
optimizer = dict(
    type="Lion",
    lr=1e-4,  # 3-10x smaller than AdamW
    betas=(0.9, 0.99),
    weight_decay=0.1,  # 3-10x larger than AdamW
)
```

### RAdam (Rectified Adam)

Rectified Adam adds variance rectification to reduce warmup sensitivity.[^radam]

**Key Characteristics:**

- Reduces the need for warmup in many cases
- Automatically adjusts the adaptive learning rate to handle high variance in early training
- Drop-in replacement for Adam with improved stability

### NAdam (Nesterov Adam)

Adam with Nesterov-style momentum for improved early training dynamics.[^nadam]

**Key Characteristics:**

- Incorporates look-ahead gradient calculation
- Can sometimes converge faster than standard Adam
- Useful when faster initial convergence is desired

### Muon (2024)

Optimizer designed for 2D weight matrices using Newton–Schulz orthogonalization.[^muon]

**Key Characteristics:**

- Typically applied selectively to specific layer types
- Uses orthogonalization for more stable updates
- Experimental but shows promise for certain architectures

---

## Learning-Rate-Free Optimizers

### D-Adaptation (2023)

D-Adaptation optimizers automatically adapt the step size, removing the need to tune learning rates.[^dadapt]

**Available variants:**

- `DAdaptAdam` - Learning-rate-free variant of Adam
- `DAdaptAdaGrad` - Learning-rate-free variant of Adagrad
- `DAdaptSGD` - Learning-rate-free variant of SGD

**Key Characteristics:**

- Automatically determines an appropriate learning rate
- Reduces hyperparameter search burden
- May require longer to converge initially while it determines the scale

---

## Memory-Efficient Optimizers

### Adafactor (2018)

Adafactor uses factored second moment estimation to dramatically reduce memory overhead.[^adafactor]

```
# Instead of storing full v matrix:
v_row = mean(v, axis=1)    # Row-wise statistics
v_col = mean(v, axis=0)    # Column-wise statistics
v ≈ outer(v_row, v_col)    # Reconstructed from factors
```

**Key Characteristics:**

- Sublinear memory cost (O(n + m) instead of O(n × m) for weight matrices)
- Particularly useful for large language models
- Available via both PyTorch (`TorchAdafactor`) and Hugging Face Transformers (`Adafactor`)

### 8-bit Optimizers (bitsandbytes)

bitsandbytes provides 8-bit quantized versions of common optimizers.[^bnb8bit]

**Key Characteristics:**

- Reduces optimizer state memory by ~4x
- Minimal impact on convergence for most tasks
- Automatically registered when `bitsandbytes` is installed
- Name collisions are prefixed with `bnb_`

---

## Distributed Training Optimizers

### DeepSpeed Optimizers

DeepSpeed provides specialized optimizers for large-scale distributed training.

**FusedAdam:** Fused GPU Adam implementation to reduce optimizer overhead.[^adam]

**DeepSpeedCPUAdam:** CPU-optimized Adam implementation for ZeRO-Offload training.[^adam]

**FusedLamb / LAMB:** Layer-wise Adaptive Moments optimizer for very large-batch training.[^lamb]

```
# LAMB scales the trust ratio per-layer:
trust_ratio = norm(params) / norm(adam_update)
params = params - learning_rate * trust_ratio * adam_update
```

**Key Characteristics:**

- LAMB enables training with batch sizes up to 64k without degradation
- Originally developed for BERT pre-training
- Communication-efficient variants (`OnebitAdam`, `OnebitLamb`) reduce distributed overhead

---

## Other Built-in Optimizers

The following optimizers are available from PyTorch's `torch.optim`:

- **ASGD:** SGD variant that maintains an average of parameters[^asgd]
- **Rprop:** Uses only the sign of gradients with per-parameter step sizes[^rprop]
- **LBFGS:** Limited-memory quasi-Newton optimizer for small/convex problems[^lbfgs]
- **Adamax:** Adam variant based on the infinity norm[^adam]
- **SparseAdam:** Adam variant for sparse gradients (e.g., embeddings)[^adam]

---

## Listing Available Optimizers

Because the available set can change depending on installed optional dependencies, you can list what your environment has registered:

```python
from visdet.engine.optim import OPTIMIZERS

print(sorted(OPTIMIZERS.module_dict.keys()))
```

!!! note
    You may see `Optimizer` in the registry output; it is the base PyTorch optimizer class and is not intended to be used as a config `type`.

[^sgd]: Robbins & Monro (1951), *A Stochastic Approximation Method*. https://doi.org/10.1214/aoms/1177729586
[^asgd]: Polyak & Juditsky (1992), *Acceleration of Stochastic Approximation by Averaging*. https://doi.org/10.1137/0330046
[^rprop]: Riedmiller & Braun (1993), *A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm*. https://doi.org/10.1109/ICNN.1993.298623
[^lbfgs]: Liu & Nocedal (1989), *On the Limited Memory BFGS Method for Large Scale Optimization*. https://doi.org/10.1007/BF01589116
[^adagrad]: Duchi, Hazan & Singer (2011), *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*. https://jmlr.org/papers/v12/duchi11a.html
[^adadelta]: Zeiler (2012), *ADADELTA: An Adaptive Learning Rate Method*. https://arxiv.org/abs/1212.5701
[^rmsprop]: Tieleman & Hinton (2012), *Lecture 6.5 — RMSProp: Divide the gradient by a running average of its recent magnitude*. https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf
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
