# MLP, Backpropagation & Optimisation
## The Engine Behind Every Neural Network

<span class="tag">Chapter 2 — Part 2</span> <span class="tag purple">Theory</span>

---

## The Multi-Layer Perceptron

An MLP with $L$ layers computes:

$$\mathbf{h}^{(0)} = \mathbf{x}$$
$$\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}$$
$$\mathbf{h}^{(\ell)} = f^{(\ell)}(\mathbf{z}^{(\ell)})$$
$$\hat{\mathbf{y}} = \mathbf{h}^{(L)}$$

- $\mathbf{W}^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$ : weight matrix of layer $\ell$
- $\mathbf{b}^{(\ell)} \in \mathbb{R}^{d_\ell}$ : bias vector
- $f^{(\ell)}$ : activation function (ReLU for hidden, softmax for output)

---

## The Chain Rule — Foundation of Backprop

For a composition $z = f(g(x))$:

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} \quad \text{where } y = g(x)$$

For a vector function $\mathbf{z} = f(\mathbf{g}(\mathbf{x}))$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{g}}{\partial \mathbf{x}}\right)^\top \frac{\partial \mathcal{L}}{\partial \mathbf{g}}$$

<div class="box">
Backpropagation is <strong>nothing more</strong> than the chain rule applied recursively from the loss back to the inputs. The "magic" is just systematic bookkeeping.
</div>

---

## Forward Pass — Concrete Example

2-layer MLP, input $\mathbf{x} \in \mathbb{R}^3$, hidden $\mathbf{h} \in \mathbb{R}^4$, output $\hat{y} \in \mathbb{R}$:

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)} \quad \in \mathbb{R}^4$$
$$\mathbf{h}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)}) \quad \in \mathbb{R}^4$$
$$z^{(2)} = \mathbf{w}^{(2)\top}\mathbf{h}^{(1)} + b^{(2)} \quad \in \mathbb{R}$$
$$\hat{y} = \sigma(z^{(2)}) \quad \in (0,1)$$
$$\mathcal{L} = -y\log\hat{y} - (1-y)\log(1-\hat{y})$$

---

## Backward Pass — Deriving the Gradients

**Step 1**: gradient of loss w.r.t. output

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

**Step 2**: through sigmoid (beautiful simplification!)

$$\frac{\partial \mathcal{L}}{\partial z^{(2)}} = \hat{y} - y$$

**Step 3**: gradient w.r.t. $\mathbf{w}^{(2)}$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}^{(2)}} = (\hat{y} - y) \cdot \mathbf{h}^{(1)}$$

**Step 4**: propagate through layer 1

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(1)}} = (\hat{y} - y) \cdot \mathbf{w}^{(2)}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(1)}} \odot \mathbf{1}[\mathbf{z}^{(1)} > 0]$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} \mathbf{x}^\top$$

---

## The General Backprop Algorithm

For layer $\ell$ (going backwards from $L$ to $1$):

$$\boldsymbol{\delta}^{(\ell)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}}$$

$$\boldsymbol{\delta}^{(\ell)} = \left(\mathbf{W}^{(\ell+1)\top} \boldsymbol{\delta}^{(\ell+1)}\right) \odot f'^{(\ell)}(\mathbf{z}^{(\ell)})$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} \mathbf{h}^{(\ell-1)\top}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}$$

> The $\odot$ symbol denotes element-wise (Hadamard) product.

---

## Gradient Descent Variants

**Batch Gradient Descent**: use all $N$ samples per update
$$\theta \leftarrow \theta - \eta \cdot \frac{1}{N}\sum_{i=1}^N \nabla_\theta \mathcal{L}_i$$
- Stable but very slow for large datasets

**Stochastic Gradient Descent (SGD)**: one sample per update
$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}_i$$
- Fast but noisy — high variance

**Mini-batch SGD**: $B$ samples per update (standard practice)
$$\theta \leftarrow \theta - \eta \cdot \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla_\theta \mathcal{L}_i$$
- Best of both worlds. Typical $B \in \{32, 64, 128, 256\}$

---

## Momentum

Plain SGD oscillates in ravines. **Momentum** accumulates a velocity vector:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta) \nabla_\theta \mathcal{L}$$
$$\theta \leftarrow \theta - \eta \mathbf{v}_t$$

- $\beta = 0.9$ is standard
- Dampens oscillations, accelerates in consistent directions
- Analogy: a ball rolling down a hill gains momentum

**Nesterov Momentum** (NAG): look ahead before computing gradient
$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta - \beta \mathbf{v}_{t-1})$$

---

## RMSProp & Adam

**RMSProp**: adapt learning rate per parameter
$$\mathbf{s}_t = \rho \mathbf{s}_{t-1} + (1-\rho)(\nabla \mathcal{L})^2$$
$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \nabla \mathcal{L}$$

**Adam** (Adaptive Moment Estimation) — the default optimiser:
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla\mathcal{L} \quad \text{(1st moment)}$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)(\nabla\mathcal{L})^2 \quad \text{(2nd moment)}$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\eta=10^{-3}$

---

## Learning Rate Scheduling

The learning rate is the **most important hyperparameter**. Common schedules:

**Step decay**: halve $\eta$ every $k$ epochs

**Cosine annealing**:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t\pi}{T}\right)$$

**Warmup + decay** (used in Transformers):
$$\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5},\ t \cdot t_{\text{warmup}}^{-1.5})$$

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)
```

---

## Regularisation — Dropout

**Dropout** (Srivastava et al., 2014): randomly zero out neurons during training

$$\tilde{\mathbf{h}} = \mathbf{h} \odot \mathbf{m}, \quad m_i \sim \text{Bernoulli}(1-p)$$

At test time: scale by $(1-p)$ (or use inverted dropout during training).

<div class="cols">
<div>

**Why it works:**
- Forces redundant representations
- Equivalent to training an **ensemble** of $2^n$ networks
- Prevents co-adaptation of neurons

</div>
<div>

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # 50% dropout
    nn.Linear(256, 10),
)
```

</div>
</div>

> Typical values: $p=0.5$ for fully-connected, $p=0.1$–$0.2$ for Transformers.

---

## Regularisation — Batch Normalisation

**BatchNorm** (Ioffe & Szegedy, 2015): normalise activations within a mini-batch

$$\hat{z}_i = \frac{z_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$
$$y_i = \gamma \hat{z}_i + \beta$$

where $\mu_\mathcal{B}$, $\sigma_\mathcal{B}^2$ are batch statistics, and $\gamma$, $\beta$ are **learnable** parameters.

**Benefits:**
- Reduces internal covariate shift
- Allows much higher learning rates
- Acts as regulariser (reduces need for dropout)
- Makes training dramatically more stable

> BatchNorm is inserted **before** the activation function (or after — still debated).

---

## Regularisation — Weight Decay (L2)

Add a penalty on large weights to the loss:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \|\mathbf{W}\|_2^2$$

Gradient update becomes:

$$\mathbf{W} \leftarrow \mathbf{W}(1 - \eta\lambda) - \eta \nabla_\mathbf{W}\mathcal{L}$$

- Shrinks weights towards zero at each step
- Prevents overfitting by limiting model complexity
- $\lambda \in [10^{-4}, 10^{-2}]$ typically

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=1e-4
)
```

---

## Weight Initialisation

Poor initialisation → vanishing or exploding gradients from the start.

**Xavier / Glorot** (for tanh/sigmoid):
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\ \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right)$$

**He / Kaiming** (for ReLU):
$$W \sim \mathcal{N}\left(0,\ \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

```python
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
nn.init.xavier_uniform_(layer.weight)
```

> PyTorch applies Kaiming initialisation by default for `nn.Linear` and `nn.Conv2d`.

---

## Full Training Loop in PyTorch

```python
model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        # 1. Forward pass
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        # 2. Zero gradients (IMPORTANT — they accumulate!)
        optimizer.zero_grad()

        # 3. Backward pass
        loss.backward()

        # 4. Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Update parameters
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
```

---

## The Vanishing Gradient Problem

In a deep network with sigmoid activations:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(L)}} \cdot \prod_{\ell=2}^{L} \mathbf{W}^{(\ell)} \cdot \text{diag}(f'(\mathbf{z}^{(\ell-1)}))$$

With sigmoid: $f'(z) \leq 0.25$. For $L=10$ layers:

$$\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}}\right\| \approx 0.25^{10} \approx 10^{-6}$$

The first layers receive **essentially zero gradient** — they don't learn.

**Solutions**: ReLU · BatchNorm · Residual connections · Careful initialisation

---

## Summary — What You Must Know

<div class="box">

1. **Backprop = chain rule** applied recursively. Derive it yourself.
2. **Adam** is the default optimiser. Understand its two moments.
3. **Dropout** prevents co-adaptation. **BatchNorm** stabilises training.
4. **Learning rate** is the most critical hyperparameter.
5. **Vanishing gradients** are the fundamental challenge of depth — every major architecture innovation addresses this.

</div>

**Next**: Convolutional Neural Networks — exploiting spatial structure.

Note: Assign exercise: implement a 3-layer MLP on MNIST from scratch (no nn.Sequential), manually implement the training loop.
