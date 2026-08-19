# Generative Models
## VAE, GAN & Diffusion — Learning to Create

<span class="tag">Chapter 7</span> <span class="tag purple">Theory</span>

---

## The Generative Modelling Problem

Given samples from an unknown distribution $p_{\text{data}}(\mathbf{x})$, learn a model $p_\theta(\mathbf{x})$ that:
- Assigns high probability to real data
- Can **generate new samples** from $p_\theta$

**Three dominant paradigms:**

| Model | Approach | Pros | Cons |
|-------|----------|------|------|
| VAE | Variational inference | Stable, latent space | Blurry samples |
| GAN | Adversarial game | Sharp samples | Training instability |
| Diffusion | Score matching | Best quality | Slow sampling |

---

## Variational Autoencoders (VAE)

**Kingma & Welling, 2013** — learn a latent representation $\mathbf{z}$ of the data.

**Generative model**: $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) d\mathbf{z}$

**Inference model** (encoder): $q_\phi(\mathbf{z} \mid \mathbf{x}) \approx p_\theta(\mathbf{z} \mid \mathbf{x})$

**ELBO** (Evidence Lower BOund) — the objective:
$$\mathcal{L}(\theta, \phi) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{regularisation}}$$

---

## The Reparameterisation Trick

The encoder outputs $\mu_\phi(\mathbf{x})$ and $\sigma_\phi(\mathbf{x})$. We need to sample $\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})$.

**Problem**: sampling is not differentiable → can't backpropagate.

**Solution**: reparameterise:
$$\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Now the randomness is in $\boldsymbol{\epsilon}$ (not a parameter) → gradients flow through $\mu_\phi$ and $\sigma_\phi$.

```python
def reparameterise(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std
```

---

## KL Divergence for Gaussians

With $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$D_{\text{KL}} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

This has a **closed form** — no Monte Carlo needed for the regularisation term.

> The KL term pushes the posterior towards the prior, ensuring the latent space is well-structured and continuous — enabling smooth interpolation.

---

## Generative Adversarial Networks (GANs)

**Goodfellow et al., 2014** — a two-player minimax game:

- **Generator** $G_\theta(\mathbf{z})$: maps noise $\mathbf{z} \sim p(\mathbf{z})$ to fake data
- **Discriminator** $D_\phi(\mathbf{x})$: distinguishes real from fake

**Minimax objective**:
$$\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]$$

At equilibrium: $D(\mathbf{x}) = \frac{1}{2}$ everywhere, $p_G = p_{\text{data}}$.

---

## GAN Training Dynamics

**Discriminator update** (maximise):
$$\mathcal{L}_D = -\mathbb{E}[\log D(\mathbf{x})] - \mathbb{E}[\log(1 - D(G(\mathbf{z})))]$$

**Generator update** (minimise, non-saturating version):
$$\mathcal{L}_G = -\mathbb{E}[\log D(G(\mathbf{z}))]$$

> The non-saturating loss is used in practice because $\log(1-D(G(\mathbf{z})))$ saturates early in training when $D$ is strong.

**Training instabilities:**
- **Mode collapse**: $G$ generates only a few modes of $p_{\text{data}}$
- **Vanishing gradients**: when $D$ is too strong
- **Oscillation**: $G$ and $D$ cycle without converging

---

## GAN Improvements

**DCGAN** (2015): convolutional architecture, BatchNorm, stable training guidelines

**Wasserstein GAN** (2017): replace JS divergence with Wasserstein distance
$$\mathcal{L} = \mathbb{E}[D(\mathbf{x})] - \mathbb{E}[D(G(\mathbf{z}))]$$
- $D$ is now a **critic** (not bounded to [0,1])
- Gradient penalty (WGAN-GP) instead of weight clipping
- Much more stable training

**Progressive GAN** (2018): grow both $G$ and $D$ progressively from low to high resolution → 1024×1024 faces

**StyleGAN** (2019): control style at different scales via adaptive instance normalisation

---

## Conditional GANs

**cGAN**: condition both $G$ and $D$ on a label $y$:
$$\min_G \max_D \mathbb{E}[\log D(\mathbf{x}, y)] + \mathbb{E}[\log(1 - D(G(\mathbf{z}, y), y))]$$

**Pix2Pix**: image-to-image translation (edges → photo, day → night)

**CycleGAN**: unpaired image translation using cycle consistency:
$$\mathcal{L}_{\text{cycle}} = \mathbb{E}[\|G_{B \to A}(G_{A \to B}(\mathbf{x})) - \mathbf{x}\|_1]$$

---

## Diffusion Models

**Ho et al., 2020** (DDPM) — the current state of the art for image generation.

**Forward process**: gradually add Gaussian noise over $T$ steps:
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

After $T$ steps: $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$

**Reverse process**: learn to denoise step by step:
$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

---

## Diffusion — Training Objective

The simplified training objective (predict the noise):

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

where $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$

The network $\boldsymbol{\epsilon}_\theta$ (typically a **U-Net**) learns to predict the noise added at step $t$.

**Sampling**: start from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, iteratively denoise using the learned reverse process.

> Diffusion models are the backbone of Stable Diffusion, DALL-E 2, and Imagen.

---

## Latent Diffusion Models

**Rombach et al., 2022** (Stable Diffusion): run diffusion in **latent space** instead of pixel space:

```
Image → VAE Encoder → Latent z → Diffusion → Latent z' → VAE Decoder → Image
```

- 8× compression → much faster training and sampling
- Text conditioning via cross-attention with CLIP text encoder
- Enables high-resolution generation on consumer hardware

---

## Comparison

| Model | Quality | Diversity | Speed | Stability |
|-------|---------|-----------|-------|-----------|
| VAE | ⭐⭐ | ⭐⭐⭐ | ⚡⚡⚡ | ✅ Stable |
| GAN | ⭐⭐⭐⭐ | ⭐⭐ | ⚡⚡⚡ | ⚠️ Tricky |
| Diffusion | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚡ | ✅ Stable |

> Diffusion models have largely superseded GANs for image generation. GANs remain competitive for real-time applications (video, 3D).

**Next**: Large Language Models — scaling everything up.

Note: Lab: train a DCGAN on MNIST/CelebA. Visualise the latent space of a VAE. Observe mode collapse.
