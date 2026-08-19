# The Transformer Architecture
## "Attention is All You Need" — Vaswani et al., 2017

<span class="tag">Chapter 5 — Part 2</span> <span class="tag purple">Theory</span>

---

## The Paper That Changed Everything

**Vaswani et al., NeurIPS 2017** — "Attention is All You Need"

- Proposed a sequence model using **only attention** — no recurrence, no convolution
- Achieved state-of-the-art on machine translation
- Enabled **massive parallelisation** during training
- Became the foundation for BERT, GPT, T5, and every modern LLM

> The Transformer is arguably the most impactful architecture in the history of machine learning.

---

## High-Level Architecture

```
Input Tokens
    ↓
[Token Embedding + Positional Encoding]
    ↓
┌─────────────────────────────┐
│  Encoder Block × N          │
│  ┌─────────────────────┐    │
│  │ Multi-Head Attention │    │
│  │ + Add & Norm         │    │
│  ├─────────────────────┤    │
│  │ Feed-Forward Network │    │
│  │ + Add & Norm         │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
    ↓
[Encoder Output]
    ↓
┌─────────────────────────────┐
│  Decoder Block × N          │
│  Masked Self-Attention       │
│  Cross-Attention             │
│  Feed-Forward                │
└─────────────────────────────┘
    ↓
Linear + Softmax → Output Tokens
```

---

## Positional Encoding

Self-attention is **permutation-invariant** — it doesn't know the order of tokens. We must inject positional information.

**Sinusoidal positional encoding** (original paper):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Properties:
- Each position has a unique encoding
- Relative positions can be computed via linear combinations
- Generalises to sequences longer than seen during training

**Learned positional embeddings** (BERT, GPT): simply learn a lookup table — works well in practice.

---

## Layer Normalisation

**LayerNorm** (Ba et al., 2016): normalise across the **feature dimension** (not batch):

$$\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu, \sigma^2$ are computed over the $d_{\text{model}}$ features of a single token.

<div class="cols">
<div>

**BatchNorm**: normalise across batch dimension
- Depends on batch size
- Problematic for variable-length sequences

</div>
<div>

**LayerNorm**: normalise across feature dimension
- Independent of batch size
- Works perfectly for sequences
- Standard in Transformers

</div>
</div>

---

## The Encoder Block

Each encoder block contains:

**1. Multi-Head Self-Attention + Residual + LayerNorm**:
$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X}))$$

**2. Position-wise Feed-Forward Network + Residual + LayerNorm**:
$$\mathbf{X}'' = \text{LayerNorm}(\mathbf{X}' + \text{FFN}(\mathbf{X}'))$$

The FFN is applied **independently to each position**:
$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

Typical dimensions: $d_{\text{model}}=512$, $d_{\text{ff}}=2048$ (4× expansion).

---

## The Decoder Block

The decoder has **three sub-layers**:

**1. Masked Multi-Head Self-Attention**: attends to previous output tokens only (causal mask)

**2. Cross-Attention**: queries from decoder, keys/values from encoder output
$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{E}, \mathbf{E}))$$

**3. Feed-Forward Network**

The **causal mask** prevents the decoder from attending to future tokens:
$$M_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$$

---

## Pre-LN vs Post-LN

**Original Transformer (Post-LN)**:
$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{Sublayer}(\mathbf{X}))$$

**Modern practice (Pre-LN)**:
$$\mathbf{X}' = \mathbf{X} + \text{Sublayer}(\text{LayerNorm}(\mathbf{X}))$$

Pre-LN is more stable during training (used in GPT-2, GPT-3, LLaMA):
- Gradients flow through the residual path without passing through LayerNorm
- Allows training without warmup in some cases

---

## Transformer Hyperparameters

| Hyperparameter | Base | Large | GPT-3 |
|---------------|------|-------|-------|
| $d_{\text{model}}$ | 512 | 1024 | 12288 |
| $d_{\text{ff}}$ | 2048 | 4096 | 49152 |
| Heads $H$ | 8 | 16 | 96 |
| Layers $N$ | 6 | 24 | 96 |
| Parameters | 65M | 340M | 175B |

---

## PyTorch: Transformer Encoder Block

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN self-attention
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x),
                                     self.norm1(x), attn_mask=mask)[0])
        # Pre-LN FFN
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
```

---

## BERT — Encoder-Only Transformer

**BERT** uses only the encoder stack, pre-trained with:

**Masked Language Modelling (MLM)**:
- Randomly mask 15% of tokens
- Predict the masked tokens
- Bidirectional context → rich representations

**Next Sentence Prediction (NSP)**:
- Predict if sentence B follows sentence A
- (Later shown to be less important — RoBERTa drops it)

```
Input:  [CLS] The cat [MASK] on the mat [SEP]
Output: [CLS] The cat  sat  on the mat [SEP]
                        ↑
                   predict this
```

---

## GPT — Decoder-Only Transformer

**GPT** uses only the decoder stack (causal/masked self-attention), pre-trained with:

**Causal Language Modelling (CLM)**:
$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})$$

Predict the next token given all previous tokens.

```
Input:  The cat sat on the
Output:     cat sat on the mat
```

> GPT is autoregressive — it generates text token by token. BERT cannot generate text (bidirectional = sees future tokens).

---

## Vision Transformer (ViT)

**Dosovitskiy et al., 2020** — apply Transformer to images:

1. Split image into $16 \times 16$ patches
2. Flatten each patch → linear projection → patch embedding
3. Add positional embeddings
4. Feed to standard Transformer encoder
5. Use [CLS] token for classification

$$\text{Image} (224 \times 224) \to 196 \text{ patches of } 16 \times 16$$

> ViT outperforms CNNs when pre-trained on large datasets (JFT-300M). On small datasets, CNNs still win — they have stronger inductive biases.

---

## Summary — The Transformer Family

| Model | Architecture | Pre-training | Best for |
|-------|-------------|-------------|---------|
| BERT | Encoder | MLM + NSP | Classification, NER, QA |
| RoBERTa | Encoder | MLM (more data) | Same + better |
| GPT-2/3 | Decoder | CLM | Generation |
| T5 | Enc-Dec | Span masking | Seq2Seq tasks |
| ViT | Encoder | Supervised/CLIP | Vision |
| CLIP | Enc+Enc | Contrastive | Vision-language |

**Next**: Reinforcement Learning — teaching agents to act.

Note: Lab: implement a mini-GPT (character-level) from scratch in PyTorch. ~100 lines of code, trains on Shakespeare in minutes.
