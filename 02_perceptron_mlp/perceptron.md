# The Perceptron & Activation Functions
## Building the Atomic Unit of Deep Learning

<span class="tag">Chapter 2 — Part 1</span> <span class="tag purple">Theory</span>

---

## The Biological Inspiration

A biological neuron:
1. Receives signals via **dendrites**
2. Integrates them in the **soma**
3. Fires an output via the **axon** if the signal exceeds a threshold

The artificial neuron mimics this:

$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top \mathbf{x} + b$$

$$\hat{y} = f(z)$$

where $f$ is the **activation function** — the key design choice.

---

## The Perceptron Learning Rule

Given a training example $(\mathbf{x}, y)$ with $y \in \{0, 1\}$:

$$w_i \leftarrow w_i + \eta \cdot (y - \hat{y}) \cdot x_i$$
$$b \leftarrow b + \eta \cdot (y - \hat{y})$$

- $\eta$ : learning rate
- $(y - \hat{y})$ : error signal

**Perceptron Convergence Theorem**: if the data is **linearly separable**, the perceptron will converge in finite steps.

<div class="box warning">
⚠️ If data is <em>not</em> linearly separable (e.g. XOR), the perceptron <strong>never converges</strong>. This was Minsky & Papert's critique.
</div>

---

## What Can a Single Neuron Learn?

A single neuron defines a **hyperplane** in input space:

$$\mathbf{w}^\top \mathbf{x} + b = 0$$

It can learn:
- AND, OR, NAND, NOR ✅
- XOR ❌ (not linearly separable)

```
AND:          XOR:
0 0 → 0       0 0 → 0
0 1 → 0       0 1 → 1  ← no single line separates these
1 0 → 0       1 0 → 1
1 1 → 1       1 1 → 0
```

> The solution: **stack multiple neurons** → Multi-Layer Perceptron.

---

## Activation Functions — Why They Matter

Without a non-linear activation, a stack of linear layers collapses to a single linear transformation:

$$\mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = \underbrace{(\mathbf{W}_2 \mathbf{W}_1)}_{\mathbf{W}'} \mathbf{x} + \mathbf{b}'$$

<div class="box">
Non-linearity is what gives neural networks their <strong>expressive power</strong>. Without it, depth is useless.
</div>

The Universal Approximation Theorem states that a single hidden layer with enough neurons and a non-linear activation can approximate **any continuous function** on a compact domain.

---

## Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

**Derivative**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

<div class="cols">
<div>

**Pros:**
- Smooth, differentiable everywhere
- Output interpretable as probability
- Used in output layer for binary classification

</div>
<div>

**Cons:**
- **Vanishing gradient**: $\sigma'(z) \leq 0.25$ always
- Output not zero-centred → slow convergence
- Saturates for large $|z|$

</div>
</div>

> In deep networks, gradients are multiplied at each layer. With sigmoid, each multiplication shrinks the gradient by ≥ 4×. After 10 layers: $0.25^{10} \approx 10^{-6}$.

---

## Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$$

**Derivative**: $\tanh'(z) = 1 - \tanh^2(z)$

- Zero-centred ✅ (better than sigmoid)
- Still saturates for large $|z|$ ❌
- Max derivative = 1 (at $z=0$) — still vanishes in deep nets

> Preferred over sigmoid for hidden layers in the pre-ReLU era.

---

## ReLU — The Game Changer

$$\text{ReLU}(z) = \max(0, z)$$

**Derivative**: $\text{ReLU}'(z) = \mathbf{1}[z > 0]$

<div class="cols">
<div>

**Pros:**
- **No vanishing gradient** for $z > 0$
- Computationally trivial
- Sparse activations (biological plausibility)
- Enabled training of very deep networks (AlexNet, 2012)

</div>
<div>

**Cons:**
- **Dying ReLU**: neurons with $z < 0$ have zero gradient — they stop learning permanently
- Not zero-centred
- Unbounded output

</div>
</div>

---

## ReLU Variants

**Leaky ReLU**: fixes dying neurons
$$\text{LReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}, \quad \alpha = 0.01$$

**ELU** (Exponential Linear Unit):
$$\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}$$

**GELU** (Gaussian Error Linear Unit) — used in BERT, GPT:
$$\text{GELU}(z) = z \cdot \Phi(z)$$
where $\Phi$ is the Gaussian CDF. Smooth approximation of ReLU.

**SiLU / Swish** — used in modern LLMs:
$$\text{SiLU}(z) = z \cdot \sigma(z)$$

---

## Softmax — Multi-class Output

For $K$-class classification, the output layer uses **softmax**:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Properties:
- Output sums to 1 → valid probability distribution
- Differentiable everywhere
- Amplifies the largest logit (winner-takes-most)

**Numerical stability trick**:
$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$$

```python
def softmax_stable(z):
    z = z - z.max(dim=-1, keepdim=True).values  # subtract max
    exp_z = torch.exp(z)
    return exp_z / exp_z.sum(dim=-1, keepdim=True)
```

---

## Loss Functions

**Binary Cross-Entropy** (sigmoid output):
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]$$

**Categorical Cross-Entropy** (softmax output):
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log \hat{y}_{ik}$$

**Mean Squared Error** (regression):
$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$

> Cross-entropy is preferred for classification because it has **steeper gradients** when the model is confidently wrong.

---

## PyTorch: Building a Neuron

```python
import torch
import torch.nn as nn

# A single neuron: linear + activation
neuron = nn.Sequential(
    nn.Linear(in_features=784, out_features=1),
    nn.Sigmoid()
)

# Forward pass
x = torch.randn(32, 784)   # batch of 32 samples
y_hat = neuron(x)           # shape: (32, 1)

# Loss
criterion = nn.BCELoss()
y = torch.randint(0, 2, (32, 1)).float()
loss = criterion(y_hat, y)

print(f"Loss: {loss.item():.4f}")
```

---

## Summary

| Activation | Range | Vanishing Grad? | Use Case |
|-----------|-------|----------------|----------|
| Sigmoid | (0,1) | ✅ Yes | Binary output |
| Tanh | (-1,1) | ✅ Yes (less) | RNN hidden states |
| ReLU | [0,∞) | ❌ No (z>0) | Default hidden layers |
| Leaky ReLU | (-∞,∞) | ❌ No | When dying ReLU is an issue |
| GELU | ≈ReLU | ❌ No | Transformers, BERT, GPT |
| Softmax | (0,1)ᴷ | — | Multi-class output |

**Next**: stacking neurons into layers, and learning how to train them with **backpropagation**.

Note: Have students implement XOR with a 2-layer network in PyTorch as exercise.
