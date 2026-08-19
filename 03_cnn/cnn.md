# Convolutional Neural Networks
## From LeNet to ResNeXt — and Beyond

<span class="tag">Chapter 3</span> <span class="tag purple">Theory</span>

---

## Why Not Just Use MLPs for Images?

A 224×224 RGB image has $224 \times 224 \times 3 = 150{,}528$ pixels.

An MLP with one hidden layer of 4096 neurons needs:

$$150{,}528 \times 4{,}096 \approx 617 \text{ million parameters}$$

**Problems:**
- Computationally intractable
- Ignores **spatial structure** — pixel (0,0) and pixel (223,223) treated identically
- Catastrophically overfits on small datasets
- Not **translation invariant** — a cat in the top-left ≠ a cat in the bottom-right

> CNNs solve all of these by exploiting the **inductive biases** of images.

---

## The Convolution Operation

A 2D convolution with kernel $\mathbf{K} \in \mathbb{R}^{k \times k}$:

$$(I * K)[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m,\ j+n] \cdot K[m,n]$$

**Key properties:**
- **Parameter sharing**: same kernel applied everywhere → translation equivariance
- **Local connectivity**: each output depends on a small receptive field
- **Sparse interactions**: far fewer parameters than fully-connected

For an input $H \times W \times C_{\text{in}}$ and $C_{\text{out}}$ filters of size $k \times k$:

$$\text{Parameters} = k^2 \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}} \text{ (biases)}$$

---

## Padding & Stride

**Padding** (same/valid):
- `same`: pad input so output has same spatial size → $\lfloor k/2 \rfloor$ zeros on each side
- `valid`: no padding → output shrinks by $k-1$

**Stride** $s$: step size of the kernel
$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$

```python
# 64 filters, 3×3 kernel, same padding
conv = nn.Conv2d(in_channels=3, out_channels=64,
                 kernel_size=3, padding=1, stride=1)
```

---

## Pooling

**Max pooling**: takes the maximum in each window
- Provides **translation invariance** (small shifts don't change output)
- Reduces spatial dimensions → fewer parameters downstream

**Average pooling**: takes the mean
- Smoother, used in modern architectures (Global Average Pooling)

**Global Average Pooling (GAP)**: average over entire spatial map
$$\text{GAP}(\mathbf{F}) = \frac{1}{H \times W} \sum_{i,j} F_{i,j}$$

> GAP replaced fully-connected layers in modern CNNs (GoogLeNet, ResNet). Fewer parameters, better generalisation, built-in spatial invariance.

---

## Receptive Field

The **receptive field** is the region of the input that influences a given output neuron.

After $L$ conv layers with kernel size $k$ and stride 1:
$$\text{RF} = 1 + L(k-1)$$

For $L=5$ layers with $k=3$: RF = $1 + 5 \times 2 = 11$

> Stacking small kernels (3×3) is more efficient than large kernels (7×7):
> - Two 3×3 layers: RF=5, params = $2 \times 9C^2 = 18C^2$
> - One 5×5 layer: RF=5, params = $25C^2$
> - Three 3×3 layers: RF=7, params = $27C^2$ vs one 7×7: $49C^2$

This insight drove VGGNet's design.

---

## LeNet-5 (1998) — The Pioneer

**LeCun et al.** — first successful CNN, deployed for cheque reading:

```
Input (32×32×1)
  → Conv(6, 5×5) → AvgPool(2×2)    → 14×14×6
  → Conv(16, 5×5) → AvgPool(2×2)   → 5×5×16
  → Flatten → FC(120) → FC(84) → FC(10)
```

- Sigmoid/tanh activations (ReLU didn't exist yet)
- ~60K parameters
- Trained on MNIST — 99%+ accuracy

> Proof of concept that CNNs work. But compute limitations prevented scaling.

---

## AlexNet (2012) — The Revolution

**Krizhevsky, Sutskever, Hinton** — ILSVRC 2012 winner:

```
Input (224×224×3)
  → Conv(96, 11×11, s=4) → MaxPool → LRN
  → Conv(256, 5×5) → MaxPool → LRN
  → Conv(384, 3×3)
  → Conv(384, 3×3)
  → Conv(256, 3×3) → MaxPool
  → FC(4096) → Dropout(0.5)
  → FC(4096) → Dropout(0.5)
  → FC(1000) → Softmax
```

**Innovations:**
- **ReLU** activations → solved vanishing gradients
- **Dropout** → regularisation
- **Data augmentation** (flips, crops, colour jitter)
- **GPU training** (2× GTX 580, model split across GPUs)
- ~60M parameters

---

## VGGNet (2014) — Depth Through Simplicity

**Simonyan & Zisserman** — ILSVRC 2014 runner-up (7.3% top-5):

**Key insight**: replace large kernels with **stacks of 3×3 convolutions**

```
VGG-16 block structure:
  [Conv(64,3×3)] × 2 → MaxPool
  [Conv(128,3×3)] × 2 → MaxPool
  [Conv(256,3×3)] × 3 → MaxPool
  [Conv(512,3×3)] × 3 → MaxPool
  [Conv(512,3×3)] × 3 → MaxPool
  FC(4096) → FC(4096) → FC(1000)
```

**Why 3×3?** Two 3×3 layers have the same receptive field as one 5×5, but:
- More non-linearities → more expressive
- Fewer parameters: $2 \times 9C^2$ vs $25C^2$

**Limits of VGG:**
- 138M parameters — enormous memory footprint
- FC layers dominate parameter count
- Training very slow
- Going deeper → vanishing gradients return

---

## The Ice Age of CNNs 🧊

Between VGG (2014) and ResNet (2015), a key observation:

> Adding more layers to VGG **hurts** performance — even on the training set.

This is **not** overfitting (training error also increases). It's the **degradation problem**:

- Deeper networks are harder to optimise
- Gradients vanish before reaching early layers
- The network cannot even learn the identity function

<div class="box warning">
⚠️ A 56-layer plain network performs <em>worse</em> than a 20-layer network on CIFAR-10. This was the crisis that ResNet solved.
</div>

---

## ResNet (2015) — Residual Learning

**He et al.** — ILSVRC 2015 winner (3.57% top-5, superhuman):

**The key idea**: instead of learning $\mathcal{H}(\mathbf{x})$, learn the **residual** $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\}) + \mathbf{x}$$

The **skip connection** (shortcut) adds the input directly to the output:

```
x ──→ [Conv → BN → ReLU → Conv → BN] ──→ (+) ──→ ReLU ──→
 └─────────────────────────────────────────┘
```

**Why does this work?**
- If the optimal function is close to identity, it's easy to push $\mathcal{F} \to 0$
- Gradients flow directly through the skip connection → no vanishing
- Enables training of 152-layer networks

---

## ResNet — Gradient Flow Analysis

With skip connections, the gradient of the loss w.r.t. an early layer:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_\ell} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \left(1 + \sum_{i=\ell}^{L-1} \frac{\partial \mathcal{F}_i}{\partial \mathbf{x}_\ell}\right)$$

The **1** term ensures gradient always flows back, regardless of the residual terms.

> ResNet is essentially an **ensemble of shallow networks** of different depths — the skip connections create exponentially many paths through the network.

---

## ResNet Variants

| Model | Layers | Params | Top-1 (ImageNet) |
|-------|--------|--------|-----------------|
| ResNet-18 | 18 | 11M | 69.8% |
| ResNet-50 | 50 | 25M | 76.1% |
| ResNet-101 | 101 | 44M | 77.4% |
| ResNet-152 | 152 | 60M | 78.3% |

**Bottleneck block** (ResNet-50+): 1×1 → 3×3 → 1×1 convolutions
- Reduces computation while maintaining expressiveness
- 1×1 convs change channel dimensions cheaply

---

## ResNeXt (2017) — Aggregated Transformations

**Xie et al.** — "Aggregated Residual Transformations"

**Idea**: instead of one wide residual block, use $C$ parallel narrower paths (cardinality):

$$\mathbf{y} = \mathbf{x} + \sum_{i=1}^{C} \mathcal{T}_i(\mathbf{x})$$

- Same complexity as ResNet-50 but higher accuracy
- Cardinality $C=32$ with width 4 outperforms ResNet-50 with width 64
- Generalises grouped convolutions

> **Key insight**: increasing cardinality is more effective than increasing depth or width.

---

## Object Detection — The Task

Beyond classification: **where** is the object, not just **what** is it?

**Classification**: single label per image
**Localisation**: single object + bounding box
**Detection**: multiple objects + bounding boxes
**Segmentation**: pixel-level labels

<div class="cols">
<div class="box">

### Two-stage detectors
1. Region proposal (RPN)
2. Classification + refinement

**R-CNN → Fast R-CNN → Faster R-CNN**
Accurate but slower

</div>
<div class="box purple">

### One-stage detectors
Direct prediction from feature map

**YOLO → SSD → RetinaNet**
Faster, slightly less accurate

</div>
</div>

---

## Faster R-CNN Architecture

```
Image
  → Backbone CNN (ResNet-50)
  → Feature Pyramid Network (FPN)
  → Region Proposal Network (RPN)
      → Anchors → objectness score + bbox regression
  → RoI Pooling / RoI Align
  → Classification head + Bbox regression head
```

**Anchor boxes**: pre-defined boxes of various scales and aspect ratios at each spatial location. The network predicts **offsets** from anchors.

**IoU (Intersection over Union)**:
$$\text{IoU} = \frac{|\text{Prediction} \cap \text{Ground Truth}|}{|\text{Prediction} \cup \text{Ground Truth}|}$$

---

## Semantic Segmentation — FCN & U-Net

**Fully Convolutional Network (FCN)**: replace FC layers with 1×1 convolutions → output is a spatial map

**U-Net** (2015) — encoder-decoder with skip connections:

```
Encoder (contracting):
  [Conv → Conv → MaxPool] × 4   ← downsampling

Bottleneck:
  Conv → Conv

Decoder (expanding):
  [UpConv → Concat(skip) → Conv → Conv] × 4

Output: 1×1 Conv → Sigmoid/Softmax
```

Skip connections concatenate encoder features to decoder → preserves fine-grained spatial detail.

> U-Net is the dominant architecture in medical image segmentation.

---

## Modern CNN Techniques

**Depthwise Separable Convolutions** (MobileNet):
- Depthwise: one filter per channel ($k^2 C$ params)
- Pointwise: 1×1 conv to mix channels ($C^2$ params)
- Total: $k^2 C + C^2$ vs $k^2 C^2$ → ~8-9× fewer params for $k=3$

**Squeeze-and-Excitation** (SENet, 2017):
- Global average pool → FC → ReLU → FC → Sigmoid
- Produces **channel attention weights** — recalibrates feature maps

**EfficientNet** (2019): compound scaling of depth, width, resolution simultaneously using a neural architecture search-derived base model.

---

## PyTorch: Building ResNet Block

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x                          # save input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual                  # skip connection
        return F.relu(out)
```

> Note: BatchNorm **before** ReLU is the original ResNet convention.

---

## Summary — The CNN Story

| Era | Model | Innovation | Why it mattered |
|-----|-------|-----------|----------------|
| 1998 | LeNet | Conv + Pool | Spatial inductive bias |
| 2012 | AlexNet | ReLU + Dropout + GPU | Scaled to ImageNet |
| 2014 | VGGNet | Deep 3×3 stacks | Simplicity + depth |
| 2015 | ResNet | Skip connections | Solved degradation |
| 2017 | ResNeXt | Cardinality | Better efficiency |
| 2019 | EfficientNet | Compound scaling | Optimal scaling |

<div class="box">
Every innovation answered a specific failure mode. <strong>Know the failure before the fix.</strong>
</div>

Note: Lab: implement ResNet-18 on CIFAR-10. Ablation study: remove skip connections and observe training collapse.
