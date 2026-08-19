# Large Language Models
## Architecture, Scale & Emergent Capabilities

<span class="tag">Chapter 8 — Part 1</span> <span class="tag purple">Theory</span>

---

## What is a Large Language Model?

An LLM is a **decoder-only Transformer** trained on massive text corpora to predict the next token:

$$P(\mathbf{x}) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1}; \theta)$$

"Large" refers to:
- **Parameters**: billions to trillions
- **Data**: hundreds of billions to trillions of tokens
- **Compute**: millions of GPU-hours

> The surprising discovery: scale alone produces **emergent capabilities** — abilities not present in smaller models and not explicitly trained for.

---

## The GPT Family

| Model | Year | Params | Tokens | Key Capability |
|-------|------|--------|--------|---------------|
| GPT-1 | 2018 | 117M | 1B | Transfer learning |
| GPT-2 | 2019 | 1.5B | 40B | Coherent long text |
| GPT-3 | 2020 | 175B | 300B | Few-shot learning |
| InstructGPT | 2022 | 175B | — | RLHF alignment |
| GPT-4 | 2023 | ~1T? | ~13T | Multimodal reasoning |

---

## Scaling Laws

**Kaplan et al., 2020** (OpenAI): loss scales as a power law with compute, parameters, and data:

$$\mathcal{L}(N) \propto N^{-\alpha_N}, \quad \mathcal{L}(D) \propto D^{-\alpha_D}, \quad \mathcal{L}(C) \propto C^{-\alpha_C}$$

**Chinchilla scaling** (Hoffmann et al., 2022): for a given compute budget $C$, the optimal model has:
$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

> GPT-3 was **undertrained** — a 70B model trained on 1.4T tokens (Chinchilla) outperforms 175B GPT-3 on most benchmarks.

---

## Modern LLM Architecture Improvements

Beyond the original Transformer, modern LLMs use:

**RoPE** (Rotary Position Embedding): encode position via rotation matrices applied to Q and K
$$\mathbf{q}_m^\top \mathbf{k}_n = f(\mathbf{q}, m)^\top f(\mathbf{k}, n) \propto g(\mathbf{q}, \mathbf{k}, m-n)$$
Relative position is encoded naturally. Used in LLaMA, Mistral, GPT-NeoX.

**SwiGLU** activation (used in LLaMA):
$$\text{SwiGLU}(\mathbf{x}) = \text{SiLU}(\mathbf{W}_1\mathbf{x}) \odot \mathbf{W}_2\mathbf{x}$$

**RMSNorm** instead of LayerNorm (faster, no mean subtraction):
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \gamma, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

---

## Grouped Query Attention (GQA)

**Problem**: Multi-Head Attention (MHA) requires storing $H$ key-value caches → memory bottleneck at inference.

**Multi-Query Attention (MQA)**: all heads share a single K and V
- Faster inference, but quality drops

**Grouped Query Attention (GQA)** (Ainslie et al., 2023): $G$ groups of heads share K and V
- Used in LLaMA-2, Mistral, Gemma
- Best trade-off between speed and quality

```
MHA:  Q₁K₁V₁  Q₂K₂V₂  Q₃K₃V₃  Q₄K₄V₄   (4 KV heads)
MQA:  Q₁K₁V₁  Q₂K₁V₁  Q₃K₁V₁  Q₄K₁V₁   (1 KV head)
GQA:  Q₁K₁V₁  Q₂K₁V₁  Q₃K₂V₂  Q₄K₂V₂   (2 KV heads)
```

---

## Tokenisation

LLMs don't operate on characters or words — they use **subword tokens**.

**Byte-Pair Encoding (BPE)**: iteratively merge the most frequent byte pairs
- GPT-2/3/4: ~50K tokens
- "unhappiness" → ["un", "happiness"] or ["un", "happy", "ness"]

**SentencePiece**: language-agnostic, handles any script

**Tiktoken** (OpenAI): fast BPE implementation

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokeniser
tokens = enc.encode("Deep learning is fascinating!")
print(tokens)        # [18953, 6975, 374, 27387, 0]
print(len(tokens))   # 5
```

---

## In-Context Learning & Emergent Abilities

GPT-3 demonstrated **few-shot learning** — solving tasks from examples in the prompt, without any gradient update:

```
Translate English to French:
sea otter → loutre de mer
peppermint → menthe poivrée
plush giraffe → girafe en peluche
cheese → ?
```

**Emergent abilities** (Wei et al., 2022): capabilities that appear abruptly at scale:
- Chain-of-thought reasoning
- Multi-step arithmetic
- Code generation
- Analogical reasoning

> These abilities are not present in 7B models but appear in 100B+ models — suggesting qualitative phase transitions.

---

## KV Cache

At inference, the Transformer recomputes K and V for all previous tokens at each step — wasteful.

**KV Cache**: store K and V for all previous tokens, only compute for the new token:

```
Step 1: compute K₁,V₁,K₂,V₂,...,Kₙ,Vₙ → cache
Step 2: compute only Kₙ₊₁,Vₙ₊₁ → append to cache
```

Memory: $2 \times T \times d_{\text{model}} \times N_{\text{layers}}$ per sequence

For LLaMA-70B with 4096 context: ~10GB per sequence → why batch size is limited at inference.

---

## Summary — LLM Design Choices

| Component | Original Transformer | Modern LLM (LLaMA-style) |
|-----------|---------------------|--------------------------|
| Position | Sinusoidal | RoPE |
| Normalisation | Post-LN | Pre-LN (RMSNorm) |
| Activation | ReLU | SwiGLU |
| Attention | MHA | GQA |
| Architecture | Enc-Dec | Decoder-only |
| Tokeniser | WordPiece | BPE (tiktoken) |

**Next**: Fine-tuning LLMs — adapting giants to specific tasks.

Note: Demo: run LLaMA-3 8B locally with llama.cpp or via HuggingFace. Show KV cache memory usage.
