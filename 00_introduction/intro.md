# 🧠 Deep Learning
## Master 2 — Artificial Intelligence

<span class="tag">2025–2026</span> <span class="tag purple">PyTorch</span> <span class="tag green">Theory + Practice</span>

> *"Deep learning is not magic — it is calculus, linear algebra, and a lot of data."*

---

## Who is this course for?

This course targets **M2 students** with:

- Solid foundations in **linear algebra** and **probability**
- Familiarity with **Python** and basic **machine learning** (regression, classification, cross-validation)
- Curiosity and willingness to go **deep into the maths**

By the end, you will be able to:

- Derive and implement backpropagation **from scratch**
- Understand and justify every architectural choice in modern networks
- Fine-tune and deploy **state-of-the-art models**
- Read and critically analyse **research papers**

---

## Course Philosophy

<div class="cols">
<div>

### 🔬 Theory first
Every concept is derived mathematically before being implemented. No black boxes.

### 🛠 PyTorch throughout
We use **PyTorch** — the de-facto standard in research. You will understand `autograd`, custom layers, training loops.

</div>
<div>

### 📖 Tell a story
Each architecture exists for a **reason**. We always ask: *what problem does this solve? What was broken before?*

### 🧪 Reproduce results
You will re-implement key papers: LeNet, ResNet, Transformer, GPT-2 (mini).

</div>
</div>

---

## Why PyTorch?

```python
import torch
import torch.nn as nn

# Everything is a tensor — differentiable by default
x = torch.randn(32, 784, requires_grad=True)
W = torch.randn(784, 256, requires_grad=True)

# Forward pass
z = x @ W                  # matrix multiply
a = torch.relu(z)          # activation

# Backward pass — one line
loss = a.sum()
loss.backward()            # autograd computes all gradients

print(W.grad.shape)        # torch.Size([784, 256])
```

> PyTorch's **dynamic computation graph** makes debugging natural — it behaves like regular Python.

---

## The Deep Learning Stack

```
┌─────────────────────────────────────────────────┐
│              Applications                        │
│   (Vision · NLP · Speech · RL · Generative)     │
├─────────────────────────────────────────────────┤
│              Architectures                       │
│   (CNN · RNN · Transformer · GAN · Diffusion)   │
├─────────────────────────────────────────────────┤
│           Training Machinery                     │
│   (Backprop · Optimisers · Regularisation)      │
├─────────────────────────────────────────────────┤
│           Mathematical Foundations               │
│   (Linear Algebra · Calculus · Probability)     │
└─────────────────────────────────────────────────┘
```

We build **bottom-up**. You cannot understand ResNet without understanding backprop. You cannot understand Transformers without understanding attention. Every layer matters.

---

## Course Roadmap

| # | Chapter | Key Concepts |
|---|---------|-------------|
| 1 | History | AI winters, breakthroughs, compute |
| 2 | Perceptron & MLP | Neurons, backprop, chain rule, optimisers |
| 3 | CNN | Convolution, pooling, ResNet, detection |
| 4 | RNN / GRU / LSTM | Sequences, vanishing gradients, gating |
| 5 | Transformers | Self-attention, BERT, ViT |
| 6 | RL | MDPs, DQN, PPO, RLHF |
| 7 | Generative | VAE, GAN, Diffusion |
| 8 | LLMs | GPT, scaling laws, fine-tuning, LoRA |

---

## Tools & Setup

```bash
# Create environment
conda create -n dl-m2 python=3.11
conda activate dl-m2

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib seaborn scikit-learn
pip install transformers datasets accelerate
pip install jupyter notebook ipywidgets
```

<div class="box warning">
⚠️ All code in this course runs on <strong>GPU</strong>. Use Google Colab (free T4) or a local CUDA setup.
</div>

---

## A Note on Mathematics

This course uses:

- **Vectors and matrices**: $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{W} \in \mathbb{R}^{m \times n}$
- **Partial derivatives**: $\frac{\partial \mathcal{L}}{\partial w_{ij}}$
- **The chain rule**: $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$
- **Probability**: $p(y \mid \mathbf{x}; \theta)$, KL divergence, entropy

> If any of these feel unfamiliar, review the **prerequisites document** before the next session.

---

## Let's Begin

<div class="box">

🎯 **Session 1 objective**: Understand *why* deep learning works, where it came from, and what makes it fundamentally different from classical ML.

</div>

Next → **Chapter 1: A Brief History of Neural Networks**

Note: Remind students to set up their environment before next session.
