# Long Short-Term Memory (LSTM)
## Explicit Memory Cells for Long-Range Dependencies

<span class="tag">Chapter 4 — Part 3</span> <span class="tag purple">Theory</span>

---

## History

**Hochreiter & Schmidhuber, 1997** — one of the most cited ML papers ever.

The LSTM was designed specifically to solve the vanishing gradient problem, which Hochreiter had analysed in his 1991 diploma thesis.

Key insight: separate the **hidden state** (short-term memory) from a **cell state** (long-term memory), with explicit gates controlling what enters, exits, and is forgotten.

> The LSTM was largely ignored until the 2010s, when GPU training made it practical. It then dominated NLP for ~5 years before Transformers.

---

## LSTM Equations

**Forget gate**: what to erase from cell state
$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input gate**: what new information to store
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

**Candidate cell state**:
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Cell state update**:
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output gate** + hidden state:
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

---

## The Cell State — The Highway

The cell state $\mathbf{c}_t$ is the key innovation:

```
c_{t-1} ──×(f_t)──→ (+) ──→ c_t ──→ tanh ──×(o_t)──→ h_t
                     ↑
                  ×(i_t)
                     ↑
                  tanh(candidate)
```

The cell state flows through time with **only element-wise operations** (multiply by forget gate, add new info). No matrix multiplication → gradients flow almost unchanged.

<div class="box">
The cell state is the LSTM's <strong>conveyor belt</strong> — information can travel across many time steps with minimal transformation.
</div>

---

## Gate Intuitions

**Forget gate** $\mathbf{f}_t$:
- $f_i \approx 1$: keep cell state dimension $i$ → remember
- $f_i \approx 0$: erase cell state dimension $i$ → forget
- Example: when processing a new sentence, forget the subject of the previous one

**Input gate** $\mathbf{i}_t$:
- Controls which new information to write to the cell
- Works together with the candidate to selectively update

**Output gate** $\mathbf{o}_t$:
- Controls what part of the cell state to expose as hidden state
- The hidden state $\mathbf{h}_t$ is a filtered version of $\mathbf{c}_t$

---

## LSTM Gradient Flow

The gradient of the loss w.r.t. the cell state at time $k$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{c}_k} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_T} \cdot \prod_{t=k+1}^{T} \mathbf{f}_t$$

This is a product of **forget gate values** — not a product of weight matrices!

- If $f_t \approx 1$ for all $t$: gradient flows unchanged → long-range learning
- If $f_t \approx 0$ at some $t$: gradient is blocked (intentional forgetting)

> The LSTM essentially learns **when to let gradients flow** — the forget gate doubles as a gradient gate.

---

## LSTM vs GRU — Architecture Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| States | $\mathbf{h}_t$ + $\mathbf{c}_t$ | $\mathbf{h}_t$ only |
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Parameters | $4(n+d) \times d$ | $3(n+d) \times d$ |
| Memory | Explicit cell state | Implicit in $\mathbf{h}_t$ |
| Expressiveness | Slightly higher | Slightly lower |
| Speed | Slower | Faster |

---

## PyTorch LSTM

```python
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=False
)

x  = torch.randn(32, 100, 128)
h0 = torch.zeros(2, 32, 256)
c0 = torch.zeros(2, 32, 256)  # initial cell state

output, (hn, cn) = lstm(x, (h0, c0))
# output: (32, 100, 256) — all hidden states
# hn:     (2, 32, 256)   — last hidden state
# cn:     (2, 32, 256)   — last cell state
```

---

## Practical Tips for LSTMs

**Initialise forget gate bias to 1**: helps the network remember by default at the start of training
```python
for name, param in lstm.named_parameters():
    if 'bias' in name:
        n = param.size(0)
        param.data[n//4:n//2].fill_(1.0)  # forget gate bias
```

**Gradient clipping** is still necessary:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

**Truncated BPTT**: for very long sequences, backpropagate only through the last $k$ steps to save memory.

---

## Applications Where LSTMs Excelled

Before Transformers, LSTMs dominated:

- **Machine translation** (Google Translate, 2016)
- **Speech recognition** (DeepSpeech)
- **Language modelling** (AWD-LSTM)
- **Named entity recognition**
- **Sentiment analysis**
- **Music generation**

> The 2016 Google Neural Machine Translation system used a stack of 8 LSTM layers with residual connections and attention — a preview of the Transformer.

---

## The Limits of LSTMs

Despite solving vanishing gradients, LSTMs still have fundamental limitations:

1. **Sequential computation**: $\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$ → cannot parallelise across time
2. **Fixed-size bottleneck**: in Seq2Seq, all information compressed into final $\mathbf{h}_T$
3. **Quadratic memory**: storing all hidden states for BPTT
4. **Still struggles** with very long dependencies (>500 steps)

> Problem 2 was addressed by the **attention mechanism** (Bahdanau, 2015). Problem 1 was addressed by the **Transformer** (Vaswani, 2017).

---

## Summary

The LSTM introduced three key ideas that remain influential:

1. **Separate memory cell** $\mathbf{c}_t$ — a dedicated long-term storage
2. **Multiplicative gating** — learnable control over information flow
3. **Additive cell update** — gradient highway through time

<div class="box">
The LSTM is not just a historical curiosity. Its gating principles reappear in Transformers (attention as soft gating), highway networks, and modern state-space models (Mamba).
</div>

**Next**: Attention mechanisms — the bridge to Transformers.

Note: Lab: sequence-to-sequence translation with LSTM + attention. Compare with and without attention mechanism.
