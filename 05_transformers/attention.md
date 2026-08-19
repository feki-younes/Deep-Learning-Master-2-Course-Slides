# Attention Mechanisms
## From Bahdanau to Self-Attention

<span class="tag">Chapter 5 — Part 1</span> <span class="tag purple">Theory</span>

---

## The Bottleneck Problem

In Seq2Seq with LSTM, the encoder compresses the entire input into a **single fixed-size vector** $\mathbf{c}$:

```
"The cat sat on the mat" → [h₁ h₂ h₃ h₄ h₅ h₆] → c = h₆
```

For long sentences, $\mathbf{c}$ must encode everything — this is a severe bottleneck.

**Empirical evidence**: translation quality degrades sharply for sentences > 20 words.

> **Bahdanau et al. (2015)**: *"We conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture."*

---

## Bahdanau Attention

**Key idea**: instead of a single context vector, compute a **different context vector for each decoder step**, as a weighted sum of all encoder hidden states.

**Alignment score** (how relevant is encoder state $j$ for decoder step $i$):
$$e_{ij} = \mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{s}_{i-1} + \mathbf{U}_a \mathbf{h}_j)$$

**Attention weights** (normalise with softmax):
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

**Context vector** (weighted sum):
$$\mathbf{c}_i = \sum_{j=1}^{T_x} \alpha_{ij} \mathbf{h}_j$$

---

## Attention as Soft Alignment

The attention weights $\alpha_{ij}$ form an **alignment matrix**:

```
         "the" "cat" "sat" "on" "the" "mat"
"le"      0.8   0.1   0.0   0.0   0.1   0.0
"chat"    0.1   0.8   0.1   0.0   0.0   0.0
"était"   0.0   0.1   0.7   0.1   0.0   0.1
"sur"     0.0   0.0   0.1   0.8   0.0   0.1
"le"      0.1   0.0   0.0   0.0   0.8   0.1
"tapis"   0.0   0.0   0.0   0.1   0.1   0.8
```

The model learns to **align** source and target words — without being told to.

> This was the first time a neural network learned interpretable word alignments automatically.

---

## Generalising Attention — Query, Key, Value

Luong (2015) and Vaswani (2017) generalised attention into the **QKV framework**:

- **Query** $\mathbf{Q}$: what am I looking for? (decoder state)
- **Key** $\mathbf{K}$: what do I have to offer? (encoder states)
- **Value** $\mathbf{V}$: what information do I actually return? (encoder states)

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

The $\sqrt{d_k}$ scaling prevents the dot products from growing too large (which would push softmax into saturation).

---

## Why Scale by $\sqrt{d_k}$?

If $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ with components $\sim \mathcal{N}(0,1)$:

$$\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$

$$\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = 0, \quad \text{Var}[\mathbf{q} \cdot \mathbf{k}] = d_k$$

So the dot product has standard deviation $\sqrt{d_k}$. Without scaling, for large $d_k$ (e.g. 512), the dot products are large → softmax saturates → gradients vanish.

Dividing by $\sqrt{d_k}$ normalises the variance back to 1.

---

## Self-Attention

In **self-attention**, queries, keys, and values all come from the **same sequence**:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

where $\mathbf{X} \in \mathbb{R}^{T \times d}$ is the input sequence.

Each token attends to **all other tokens** in the sequence:

$$\text{SelfAttn}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{X}\mathbf{W}^Q(\mathbf{X}\mathbf{W}^K)^\top}{\sqrt{d_k}}\right)\mathbf{X}\mathbf{W}^V$$

> This allows direct modelling of dependencies between any two positions — regardless of distance. No more vanishing gradients over long sequences.

---

## Multi-Head Attention

A single attention head may focus on one type of relationship. **Multi-head attention** runs $H$ attention heads in parallel:

$$\text{head}_h = \text{Attention}(\mathbf{X}\mathbf{W}_h^Q, \mathbf{X}\mathbf{W}_h^K, \mathbf{X}\mathbf{W}_h^V)$$

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\mathbf{W}^O$$

- Each head uses $d_k = d_{\text{model}} / H$ dimensions
- Different heads learn different types of relationships:
  - Syntactic dependencies
  - Coreference resolution
  - Positional patterns

> With $H=8$ heads and $d_{\text{model}}=512$: each head has $d_k=64$.

---

## Complexity: Attention vs RNN

| | Self-Attention | RNN |
|--|---------------|-----|
| Complexity per layer | $O(T^2 d)$ | $O(T d^2)$ |
| Sequential operations | $O(1)$ | $O(T)$ |
| Max path length | $O(1)$ | $O(T)$ |
| Parallelisable | ✅ Yes | ❌ No |

Self-attention is **quadratic in sequence length** — a bottleneck for very long sequences (documents, audio). This motivated sparse attention, linear attention, and state-space models.

But for typical NLP sequences (< 512 tokens), the parallelism advantage is decisive.

---

## PyTorch: Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq_len, d_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)   # (batch, heads, seq, seq)
    return torch.matmul(weights, V), weights

# PyTorch 2.0+ has a fused, memory-efficient version:
# F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
```

---

## Embedding Algorithms — Word2Vec

Before contextual embeddings, **static embeddings** mapped words to vectors:

**Word2Vec** (Mikolov et al., 2013):
- **CBOW**: predict centre word from context
- **Skip-gram**: predict context from centre word

$$\mathcal{L} = -\log \sigma(\mathbf{v}_{w_O}^\top \mathbf{v}_{w_I}) - \sum_{k=1}^{K} \log \sigma(-\mathbf{v}_{w_k}^\top \mathbf{v}_{w_I})$$

Properties: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$

**GloVe** (Pennington et al., 2014): factorises the word co-occurrence matrix.

> Static embeddings: one vector per word, regardless of context. "Bank" has the same vector in "river bank" and "bank account".

---

## Contextual Embeddings — BERT

**BERT** (Devlin et al., 2018): Bidirectional Encoder Representations from Transformers

- Pre-trained on **Masked Language Modelling** (MLM): predict masked tokens
- And **Next Sentence Prediction** (NSP)
- Fine-tuned on downstream tasks

$$\text{BERT}(\text{"The [MASK] sat on the mat"}) \to \text{"cat"}$$

The embedding of "bank" in "river bank" ≠ "bank" in "bank account" — **context-dependent**.

---

## Sentence-BERT

**Sentence-BERT** (Reimers & Gurevych, 2019): fine-tunes BERT with **siamese networks** for sentence similarity:

```
Sentence A → BERT → Pool → u
Sentence B → BERT → Pool → v
Similarity = cosine(u, v)
```

Training objective (natural language inference):
$$\mathcal{L} = \text{softmax}(|u - v|, u \odot v, u, v)$$

Applications: semantic search, clustering, RAG retrieval.

> Standard BERT is not suitable for sentence embeddings — the [CLS] token embedding is a poor sentence representation without fine-tuning.

---

## Summary

| Method | Type | Context-aware? | Use case |
|--------|------|---------------|----------|
| Word2Vec | Static | ❌ | Fast baseline |
| GloVe | Static | ❌ | Pre-2018 NLP |
| BERT | Contextual | ✅ | Classification, NER, QA |
| Sentence-BERT | Contextual | ✅ | Semantic similarity, RAG |

**Next**: The full Transformer architecture — encoder, decoder, positional encoding.

Note: Demo: visualise attention heads in BERT using BertViz. Show how different heads capture different linguistic phenomena.
