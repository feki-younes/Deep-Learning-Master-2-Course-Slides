
# Fine-Tuning Large Language Models
## LoRA, RLHF, Instruction Tuning & Beyond

<span class="tag">Chapter 8 — Part 2</span> <span class="tag green">Theory + Code</span>

---

## Why Fine-Tune?

A pre-trained LLM knows a lot — but it doesn't know:
- Your specific domain (medical, legal, financial)
- How to follow instructions politely
- Your company's tone and format
- How to refuse harmful requests

**Fine-tuning** adapts the model to a specific task or behaviour using a smaller, curated dataset.

**Challenge**: fine-tuning a 70B model requires ~140GB of GPU memory (fp16) — impractical for most.

---

## Full Fine-Tuning

Update **all** parameters of the pre-trained model:

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
# All ~8B parameters are trainable
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

**Pros**: maximum flexibility, best performance
**Cons**: requires enormous compute and memory, risk of catastrophic forgetting

> For most use cases, full fine-tuning is overkill. Parameter-efficient methods achieve 95%+ of the performance at 1% of the cost.

---

## LoRA — Low-Rank Adaptation

**Hu et al., 2021** — the dominant PEFT method.

**Key insight**: the weight updates during fine-tuning have **low intrinsic rank**.

Instead of updating $\mathbf{W} \in \mathbb{R}^{d \times k}$ directly, decompose the update:

$$\mathbf{W}' = \mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$$

where $\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

- $\mathbf{W}_0$ is **frozen** — no gradient computed
- Only $\mathbf{A}$ and $\mathbf{B}$ are trained
- Initialisation: $\mathbf{A} \sim \mathcal{N}(0, \sigma^2)$, $\mathbf{B} = \mathbf{0}$ → $\Delta\mathbf{W} = 0$ at start

---

## LoRA — Parameter Savings

For a weight matrix $\mathbf{W} \in \mathbb{R}^{4096 \times 4096}$ with rank $r=16$:

| Method | Trainable params |
|--------|----------------|
| Full fine-tuning | $4096^2 = 16.8M$ |
| LoRA ($r=16$) | $2 \times 4096 \times 16 = 131K$ |
| **Reduction** | **128×** |

Applied to all attention matrices (Q, K, V, O) in a 7B model:
- Full: 7B parameters
- LoRA ($r=16$): ~4M parameters → **0.06% of total**

---

## LoRA in PyTorch (with PEFT)

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # rank
    lora_alpha=32,                 # scaling factor
    target_modules=["q_proj", "v_proj"],  # which layers
    lora_dropout=0.05,
    bias="none",
)

# Wrap model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920
# trainable%: 0.0622%
```

---

## QLoRA — Quantised LoRA

**Dettmers et al., 2023** — fine-tune a 65B model on a single 48GB GPU:

1. **Quantise** the base model to **4-bit NF4** (NormalFloat4)
2. Apply LoRA adapters in **bf16**
3. Use **double quantisation** to further reduce memory
4. Use **paged optimisers** to handle memory spikes

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

## Instruction Tuning

**Goal**: teach the model to follow natural language instructions.

**Dataset format** (Alpaca-style):
```json
{
  "instruction": "Summarise the following text in one sentence.",
  "input": "Deep learning is a subset of machine learning...",
  "output": "Deep learning uses multi-layer neural networks to learn representations from data."
}
```

**Training**: standard causal LM loss, but only on the **output** tokens (mask the instruction).

Key datasets: Alpaca (52K), FLAN (1.8K tasks), OpenHermes, ShareGPT.

> Instruction tuning transforms a "text completion" model into an "assistant" model.

---

## RLHF — Reinforcement Learning from Human Feedback

**Three-stage pipeline** (InstructGPT, Ouyang et al., 2022):

**Stage 1 — SFT**: fine-tune on human demonstrations
```
Prompt: "Explain quantum entanglement to a 10-year-old"
Response: [human-written high-quality answer]
```

**Stage 2 — Reward Model**: train on human preference pairs
```
Response A: [good] vs Response B: [bad]
→ r_φ(prompt, response) predicts human preference score
```

**Stage 3 — PPO**: optimise policy with reward model
$$\mathcal{L} = \mathbb{E}[r_\phi(x,y)] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{SFT}})$$

---

## DPO — Direct Preference Optimisation

**Rafailov et al., 2023** — RLHF without the RL:

Directly optimise on preference pairs $(y_w, y_l)$ (winner, loser):

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**Advantages over RLHF:**
- No reward model needed
- No PPO training loop
- More stable, simpler to implement
- Competitive performance

> DPO has largely replaced RLHF for alignment in open-source models (Zephyr, Tulu, OpenHermes).

---

## Retrieval-Augmented Generation (RAG)

Fine-tuning is expensive. For knowledge-intensive tasks, **RAG** is often better:

```
Query → Embed → Vector DB → Top-k chunks
                                ↓
Query + Context → LLM → Answer
```

**Components:**
- **Embedding model**: Sentence-BERT, BGE, E5
- **Vector database**: FAISS, Chroma, Qdrant, Weaviate
- **LLM**: any instruction-tuned model

> RAG allows the model to access up-to-date information without retraining. Fine-tuning teaches *how* to behave; RAG provides *what* to say.

---

## Evaluation of LLMs

**Perplexity**: intrinsic measure of language model quality
$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log P(x_t \mid x_{<t})\right)$$

**Benchmarks:**
- **MMLU**: 57-subject multiple choice (knowledge)
- **HumanEval**: code generation (pass@k)
- **GSM8K**: grade school math (chain-of-thought)
- **HellaSwag**: commonsense reasoning
- **MT-Bench**: multi-turn conversation quality

<div class="box warning">
⚠️ Benchmark contamination is a serious issue — models may have seen test sets during pre-training. Always evaluate on held-out data.
</div>

---

## The Full Pipeline — From Pre-training to Deployment

```
1. Pre-training
   Massive corpus (Common Crawl, Books, Code)
   → Causal LM objective
   → Base model (knows language, facts, code)

2. Instruction Tuning (SFT)
   Curated instruction-response pairs
   → Follows instructions

3. Alignment (RLHF / DPO)
   Human preference data
   → Helpful, harmless, honest

4. Quantisation (GPTQ / AWQ / GGUF)
   Reduce to 4-8 bit
   → Deployable on consumer hardware

5. Deployment
   vLLM / llama.cpp / TGI
   → Serve at scale
```

---

## Summary — Fine-Tuning Decision Tree

<div class="box">

**Do you need domain knowledge?** → RAG first, fine-tune if RAG is insufficient

**Do you need behaviour change?** → Instruction tuning (SFT)

**Do you need alignment?** → DPO on preference pairs

**Limited GPU memory?** → QLoRA (4-bit base + LoRA adapters)

**Production deployment?** → Quantise (GPTQ/AWQ) + vLLM

</div>

---

## Congratulations 🎉

You have completed the **Deep Learning M2** course.

You now understand:
- The mathematical foundations (backprop, chain rule, optimisation)
- Every major architecture (CNN, RNN, Transformer, GAN, Diffusion)
- How LLMs are built, trained, and fine-tuned
- The historical context and design decisions behind each choice

> *"The best way to understand deep learning is to implement it. Go build something."*

Note: Final project: fine-tune LLaMA-3 8B with QLoRA on a domain-specific dataset. Evaluate with ROUGE/BERTScore. Present results.
