# Recurrent Neural Networks
## Modelling Sequences — and Why It's Hard

<span class="tag">Chapter 4 — Part 1</span> <span class="tag purple">Theory</span>

---

## The Problem with MLPs for Sequences

An MLP maps a **fixed-size input** to a **fixed-size output**. But many real problems involve sequences:

- Text: variable-length sentences
- Speech: variable-length audio
- Time series: stock prices, sensor data
- Video: variable number of frames

**What we need:**
- Process inputs of **arbitrary length**
- Maintain **memory** of past inputs
- Share parameters across time steps

> CNNs can handle local patterns in sequences (1D conv), but struggle with **long-range dependencies**.

---

## The RNN Formulation

At each time step $t$, an RNN computes:

$$\mathbf{h}_t = f(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$$
$$\mathbf{y}_t = \mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y$$

- $\mathbf{h}_t \in \mathbb{R}^d$ : hidden state (memory)
- $\mathbf{x}_t \in \mathbb{R}^n$ : input at time $t$
- $\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{W}_{hy}$ : **shared** across all time steps

The same parameters are reused at every step — this is **parameter sharing in time**.

---

## Unrolling the RNN

An RNN "unrolled" over $T$ steps looks like a very deep network:

```
x₁ → [h₁] → y₁
x₂ → [h₂] → y₂
x₃ → [h₃] → y₃
...
xT → [hT] → yT

h₀ → h₁ → h₂ → h₃ → ... → hT
```

Each arrow between hidden states uses the **same** $\mathbf{W}_{hh}$.

> This is why RNNs are powerful — and why they're hard to train.

---

## Backpropagation Through Time (BPTT)

To train an RNN, we unroll it and apply backprop:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}$$

The gradient at step $t$ flows back through all previous steps:

$$\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_k} = \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \cdot \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}}$$

Each factor $\frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \mathbf{W}_{hh}^\top \cdot \text{diag}(f'(\mathbf{z}_{i-1}))$

---

## The Vanishing & Exploding Gradient Problem

The product of $T$ Jacobians:

$$\prod_{i=k+1}^{t} \mathbf{W}_{hh}^\top \cdot \text{diag}(f'(\mathbf{z}_{i-1}))$$

If the largest singular value of $\mathbf{W}_{hh}$ is $\lambda$:
- $\lambda < 1$ → gradients **vanish** exponentially → early steps don't learn
- $\lambda > 1$ → gradients **explode** → training diverges

**Practical consequences:**
- RNNs struggle to learn dependencies longer than ~10-20 steps
- The network "forgets" early inputs

**Mitigation:**
- Gradient clipping (for exploding): $\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\tau}{\|\mathbf{g}\|}$ if $\|\mathbf{g}\| > \tau$
- Gated architectures (for vanishing): GRU, LSTM

---

## RNN Architectures

**Many-to-one**: sentiment analysis, document classification
```
x₁ x₂ x₃ x₄ → hT → y
```

**One-to-many**: image captioning
```
x → h₁ h₂ h₃ h₄ → y₁ y₂ y₃ y₄
```

**Many-to-many (same length)**: POS tagging, NER
```
x₁ x₂ x₃ → y₁ y₂ y₃
```

**Seq2Seq (encoder-decoder)**: machine translation
```
Encoder: x₁ x₂ x₃ → context vector c
Decoder: c → y₁ y₂ y₃ y₄
```

---

## Bidirectional RNNs

A standard RNN only sees **past** context. For tasks like NER, future context matters too.

**BiRNN**: run two RNNs — one forward, one backward — and concatenate hidden states:

$$\overrightarrow{\mathbf{h}}_t = f(\mathbf{W}\overrightarrow{\mathbf{h}}_{t-1} + \mathbf{U}\mathbf{x}_t)$$
$$\overleftarrow{\mathbf{h}}_t = f(\mathbf{W}\overleftarrow{\mathbf{h}}_{t+1} + \mathbf{U}\mathbf{x}_t)$$
$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t;\ \overleftarrow{\mathbf{h}}_t]$$

> Cannot be used for **autoregressive generation** (future is unknown at inference time). Used in BERT-style encoders.

---

## PyTorch RNN

```python
import torch.nn as nn

rnn = nn.RNN(
    input_size=128,    # dimension of x_t
    hidden_size=256,   # dimension of h_t
    num_layers=2,      # stacked RNN
    batch_first=True,  # input shape: (batch, seq, features)
    dropout=0.3,
    bidirectional=False
)

# x: (batch=32, seq_len=50, input_size=128)
x = torch.randn(32, 50, 128)
h0 = torch.zeros(2, 32, 256)  # (num_layers, batch, hidden)

output, hn = rnn(x, h0)
# output: (32, 50, 256) — all hidden states
# hn:     (2, 32, 256)  — last hidden state
```

---

## Limitations of Vanilla RNN

| Issue | Description |
|-------|-------------|
| Vanishing gradients | Cannot learn long-range dependencies |
| Exploding gradients | Training instability (mitigated by clipping) |
| Sequential computation | Cannot parallelise across time steps |
| Fixed-size context | All information compressed into $\mathbf{h}_t$ |

> These limitations motivated GRU (2014) and LSTM (1997/2014), and ultimately the Transformer (2017).

**Next**: GRU — a simpler gated solution.

Note: Exercise: implement character-level RNN language model on Shakespeare text.
