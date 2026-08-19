# Gated Recurrent Units (GRU)
## Learning What to Remember and What to Forget

<span class="tag">Chapter 4 — Part 2</span> <span class="tag purple">Theory</span>

---

## Motivation — Gating the Information Flow

The vanishing gradient problem in RNNs stems from the fact that **all information** must pass through the same transformation at every step.

**Key insight**: what if the network could **selectively** decide:
- How much of the past to remember?
- How much of the new input to incorporate?

This is the idea behind **gating mechanisms** — learnable switches that control information flow.

---

## GRU Equations

**Cho et al., 2014** — introduced as part of the Seq2Seq paper:

**Reset gate** $\mathbf{r}_t$: how much past to forget
$$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r)$$

**Update gate** $\mathbf{z}_t$: how much to update the hidden state
$$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z)$$

**Candidate hidden state**:
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h)$$

**New hidden state** (interpolation):
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

---

## Intuition Behind the Gates

**Update gate** $\mathbf{z}_t \in (0,1)^d$:
- $z_i \approx 1$: replace $h_{t-1,i}$ with new candidate → **update**
- $z_i \approx 0$: keep $h_{t-1,i}$ unchanged → **remember**

**Reset gate** $\mathbf{r}_t \in (0,1)^d$:
- $r_i \approx 1$: use full past hidden state for candidate
- $r_i \approx 0$: ignore past → compute candidate from input only

<div class="box">
When $\mathbf{z}_t \approx 0$ everywhere, the GRU simply copies the previous hidden state — the gradient flows through unchanged. This is the key to avoiding vanishing gradients.
</div>

---

## GRU vs Vanilla RNN — Gradient Flow

In a vanilla RNN:
$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \mathbf{W}_{hh}^\top \cdot \text{diag}(f'(\mathbf{z}_{t-1}))$$

In a GRU, when $\mathbf{z}_t \approx 0$:
$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} \approx \mathbf{I} - \text{diag}(\mathbf{z}_t) \approx \mathbf{I}$$

The gradient flows through an **identity-like** transformation → no vanishing.

> The GRU essentially learns to create **gradient highways** for important information.

---

## Parameter Count

For hidden size $d$ and input size $n$:

| Component | Parameters |
|-----------|-----------|
| Reset gate $\mathbf{W}_r, \mathbf{U}_r$ | $(n+d) \times d$ |
| Update gate $\mathbf{W}_z, \mathbf{U}_z$ | $(n+d) \times d$ |
| Candidate $\mathbf{W}_h, \mathbf{U}_h$ | $(n+d) \times d$ |
| **Total** | $3(n+d) \times d$ |

Compare to vanilla RNN: $(n+d) \times d$ — GRU has **3× more parameters**.

---

## PyTorch GRU

```python
gru = nn.GRU(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=True
)

x  = torch.randn(32, 50, 128)
h0 = torch.zeros(4, 32, 256)  # 2 layers × 2 directions

output, hn = gru(x, h0)
# output: (32, 50, 512)  — bidirectional: 256×2
# hn:     (4, 32, 256)
```

---

## When to Use GRU

<div class="cols">
<div class="box">

### GRU is preferred when:
- Smaller dataset (fewer params → less overfitting)
- Faster training is needed
- Simpler architecture is desired
- Performance is comparable to LSTM

</div>
<div class="box purple">

### LSTM is preferred when:
- Very long sequences
- Tasks requiring fine-grained memory control
- When the extra cell state helps (e.g. language modelling)

</div>
</div>

> In practice, GRU and LSTM perform similarly on most tasks. GRU is faster and simpler. The Transformer has largely superseded both for NLP.

---

## Summary

The GRU solves the vanishing gradient problem through two gates:

1. **Reset gate**: controls how much past to use when computing the candidate
2. **Update gate**: interpolates between past state and new candidate

The interpolation formula $\mathbf{h}_t = (1-\mathbf{z}_t)\mathbf{h}_{t-1} + \mathbf{z}_t\tilde{\mathbf{h}}_t$ is the key — it creates a direct gradient path when $\mathbf{z}_t$ is small.

**Next**: LSTM — the original gated architecture, with an explicit memory cell.
