# A Brief History of Neural Networks
## From McCulloch-Pitts (1943) to ChatGPT (2022)

<span class="tag history">History</span> <span class="tag">Chapter 1</span>

> *"Every time someone says neural networks are dead, they come back stronger."*

---

## Why Study History?

Understanding **where we came from** explains:

- Why certain architectures look the way they do
- Why some ideas were abandoned — and then rediscovered
- What **actually** caused the breakthroughs (hint: it's not just algorithms)
- Why the field moves in **cycles** of hype and winter

<div class="box">
The history of deep learning is the history of three ingredients coming together at the right time: <strong>algorithms</strong>, <strong>data</strong>, and <strong>compute</strong>.
</div>

---

## 1943 — The First Neuron

**McCulloch & Pitts** proposed the first mathematical model of a neuron:

$$y = \begin{cases} 1 & \text{if } \sum_i w_i x_i \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

- Binary inputs and outputs
- Fixed weights (no learning)
- Showed that networks of such neurons could compute **any logical function**

> This was purely theoretical — no learning algorithm existed yet.

---

## 1958 — The Perceptron

**Frank Rosenblatt** built the **Mark I Perceptron** — a physical machine that could learn:

$$w_i \leftarrow w_i + \eta \cdot (y - \hat{y}) \cdot x_i$$

- First **learning rule** for a neural network
- Demonstrated on image classification (letters)
- The New York Times called it *"the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself"*

<div class="box warning">
⚠️ The hype was enormous — and premature.
</div>

---

## 1969 — The First AI Winter

**Minsky & Papert** published *Perceptrons* (1969):

- Proved the perceptron **cannot learn XOR** (not linearly separable)
- Argued that multi-layer networks were intractable
- Funding dried up — **first AI winter** begins

```
XOR truth table:
  x1=0, x2=0 → 0
  x1=0, x2=1 → 1   ← not linearly separable
  x1=1, x2=0 → 1
  x1=1, x2=1 → 0
```

> The critique was valid for single-layer networks. Multi-layer networks *could* solve XOR — but no one knew how to train them yet.

---

## 1986 — Backpropagation

**Rumelhart, Hinton & Williams** popularised **backpropagation** in their landmark *Nature* paper:

- Efficient algorithm to compute gradients in multi-layer networks
- Made training deep networks **theoretically possible**
- Demonstrated on XOR, word embeddings, encoder-decoder tasks

<div class="box">
Backpropagation is just the <strong>chain rule of calculus</strong>, applied systematically to a computation graph. We will derive it fully in Chapter 2.
</div>

> Note: the algorithm was independently discovered by Werbos (1974) in his PhD thesis — largely ignored at the time.

---

## 1989–1998 — Early Successes & Limits

**Yann LeCun** applied backprop to convolutional networks:

- **LeNet-5** (1998): reads handwritten digits on cheques for US banks
- Introduced convolution, pooling, end-to-end training

But by the mid-1990s, **SVMs** and other kernel methods outperformed neural networks on most benchmarks.

Reasons neural nets struggled:
- **Vanishing gradients** in deep networks
- **Insufficient data** — ImageNet didn't exist yet
- **Insufficient compute** — no GPUs for ML

> Second AI winter: neural networks fall out of fashion. Hinton, LeCun, Bengio keep working in obscurity.

---

## 2006 — The Deep Learning Renaissance

**Hinton & Salakhutdinov** (Science, 2006):

- Introduced **Deep Belief Networks** with greedy layer-wise pre-training
- Showed deep networks *could* be trained if initialised properly
- Coined the term **"deep learning"**

Simultaneously:
- **Bengio** worked on unsupervised pre-training
- **LeCun** continued pushing convolutional nets

<div class="box">
The key insight: <em>initialisation matters enormously</em>. Random initialisation leads to vanishing gradients. Smart initialisation (or pre-training) unlocks depth.
</div>

---

## 2012 — The ImageNet Moment 🚀

**AlexNet** (Krizhevsky, Sutskever, Hinton) — ILSVRC 2012:

| Model | Top-5 Error |
|-------|------------|
| Best non-DL (2011) | 25.8% |
| **AlexNet (2012)** | **16.4%** |
| Human estimate | ~5% |

What made AlexNet work:
- **ReLU** activations (solved vanishing gradients)
- **Dropout** regularisation
- **Data augmentation**
- Training on **2× NVIDIA GTX 580 GPUs**

> This single result changed the field overnight. Every major tech company pivoted to deep learning within 12 months.

---

## 2012–2017 — The CNN Era

Rapid progress on ImageNet:

| Year | Model | Top-5 Error | Key Innovation |
|------|-------|------------|----------------|
| 2012 | AlexNet | 16.4% | ReLU, Dropout, GPU |
| 2014 | VGGNet | 7.3% | Very deep, 3×3 convs |
| 2014 | GoogLeNet | 6.7% | Inception modules |
| 2015 | **ResNet** | **3.57%** | Residual connections |
| 2017 | SENet | 2.25% | Channel attention |

> ResNet surpassed **human-level performance** (5%) on ImageNet.

---

## 2014–2017 — Beyond Vision

The CNN era also saw breakthroughs in other domains:

- **2013**: Word2Vec — dense word embeddings
- **2014**: Seq2Seq — neural machine translation
- **2014**: GANs — generative adversarial networks (Goodfellow)
- **2015**: Attention mechanism (Bahdanau)
- **2016**: AlphaGo defeats Lee Sedol
- **2017**: **"Attention is All You Need"** — the Transformer

<div class="box">
The Transformer paper is arguably the most impactful ML paper of the decade. It made RNNs largely obsolete and enabled the LLM revolution.
</div>

---

## 2018–2022 — The Language Model Era

| Year | Model | Parameters | Key Contribution |
|------|-------|-----------|-----------------|
| 2018 | BERT | 340M | Bidirectional pre-training |
| 2019 | GPT-2 | 1.5B | Autoregressive generation |
| 2020 | GPT-3 | 175B | Few-shot learning |
| 2021 | CLIP | 400M | Vision-language alignment |
| 2022 | ChatGPT | ~175B | RLHF instruction tuning |
| 2023 | GPT-4 | ~1T? | Multimodal reasoning |

> **Scaling laws** (Kaplan et al., 2020): performance improves predictably with more parameters, data, and compute.

---

## The Three Ingredients

Every breakthrough in deep learning history traces back to improvements in:

<div class="cols">
<div class="box">

### 📊 Data
- ImageNet (1.2M images)
- Common Crawl (petabytes of text)
- Without data, algorithms are useless

</div>
<div class="box purple">

### ⚙️ Compute
- GPU → TPU → GPU clusters
- AlexNet: 2 GPUs, 6 days
- GPT-3: ~$4.6M in compute

</div>
</div>

<div class="box" style="margin-top:1em">

### 🧮 Algorithms
ReLU · Dropout · BatchNorm · Residual connections · Attention · Adam optimiser

</div>

> The algorithms were often known for years. What changed was **scale**.

---

## Lessons from History

1. **Ideas recycle** — attention was proposed in 2015, transformers in 2017, but the core idea traces to 1990s memory models
2. **Compute is the hidden variable** — many "failed" ideas from the 1990s work perfectly today with more compute
3. **Empiricism drives theory** — practitioners discover what works; theorists explain it later
4. **Hype is dangerous** — both AI winters were preceded by overclaiming

<div class="box warning">
⚠️ We may be in a third hype cycle. Critical thinking about capabilities and limitations is essential.
</div>

---

## Next Chapter

We now have the context. Time to build from the ground up.

**Chapter 2** starts with the simplest possible learning unit — the **Perceptron** — and derives everything from first principles.

> *"If you can't derive it, you don't understand it."*

Note: Ask students: what do you think will be the next breakthrough? Data? Compute? A new algorithm?
