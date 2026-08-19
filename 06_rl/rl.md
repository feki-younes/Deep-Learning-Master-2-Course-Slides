# Deep Reinforcement Learning
## Teaching Agents to Act Through Experience

<span class="tag">Chapter 6</span> <span class="tag purple">Theory</span>

---

## What is Reinforcement Learning?

An **agent** interacts with an **environment** to maximise cumulative **reward**:

```
Agent ──action aₜ──→ Environment
  ↑                       ↓
  └── reward rₜ, state sₜ₊₁ ──┘
```

Unlike supervised learning: **no labelled examples** — only a reward signal.

**Key challenge**: the **credit assignment problem** — which past actions caused the current reward?

> RL is how AlphaGo, ChatGPT (RLHF), and game-playing agents are trained.

---

## Markov Decision Processes (MDPs)

Formally, RL is defined as an MDP: $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$

- $\mathcal{S}$: state space
- $\mathcal{A}$: action space
- $\mathcal{P}(s' \mid s, a)$: transition probability
- $\mathcal{R}(s, a)$: reward function
- $\gamma \in [0,1)$: discount factor

**Markov property**: $P(s_{t+1} \mid s_t, a_t, s_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$

The future depends only on the **current state**, not the history.

---

## Value Functions

**State-value function** $V^\pi(s)$: expected cumulative reward from state $s$ under policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]$$

**Action-value function** $Q^\pi(s,a)$: expected return from state $s$, taking action $a$, then following $\pi$:

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0=s, a_0=a\right]$$

**Bellman equation**:
$$Q^\pi(s,a) = \mathcal{R}(s,a) + \gamma \sum_{s'} \mathcal{P}(s'\mid s,a) V^\pi(s')$$

---

## Q-Learning

**Q-learning** (Watkins, 1989): model-free, off-policy algorithm

Update rule (temporal difference):
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]$$

The term in brackets is the **TD error** $\delta_t$:
$$\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$$

**Convergence**: Q-learning converges to $Q^*$ if all state-action pairs are visited infinitely often and the learning rate satisfies Robbins-Monro conditions.

---

## Deep Q-Network (DQN)

**Mnih et al., 2015** (DeepMind) — plays Atari games from raw pixels:

Replace the Q-table with a **neural network** $Q(s, a; \theta)$:

```
Pixels (84×84×4) → Conv → Conv → Conv → FC → FC → Q-values
```

**Two key tricks** that make training stable:

**1. Experience Replay**: store transitions $(s,a,r,s')$ in a buffer, sample random mini-batches
- Breaks temporal correlations
- Reuses data efficiently

**2. Target Network**: use a separate, slowly-updated network for TD targets
$$\mathcal{L} = \mathbb{E}\left[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2\right]$$

---

## Policy Gradient Methods

Instead of learning $Q$, directly optimise the **policy** $\pi_\theta(a \mid s)$:

$$\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t r_t\right]$$

**Policy Gradient Theorem**:
$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]$$

where $G_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ is the return from step $t$.

**REINFORCE algorithm**: Monte Carlo estimate of the policy gradient.

---

## Actor-Critic Methods

**Problem with REINFORCE**: high variance — $G_t$ fluctuates a lot.

**Solution**: subtract a **baseline** (the value function):

$$\nabla_\theta \mathcal{J} = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \underbrace{(G_t - V(s_t))}_{\text{advantage } A_t}\right]$$

**Actor-Critic**: two networks
- **Actor** $\pi_\theta(a \mid s)$: the policy
- **Critic** $V_\phi(s)$: estimates the value function

The advantage $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ has much lower variance than $G_t$.

---

## PPO — Proximal Policy Optimisation

**Schulman et al., 2017** — the dominant RL algorithm today (used in ChatGPT/RLHF):

**Problem**: large policy updates can be catastrophically bad.

**Solution**: clip the policy ratio to stay close to the old policy:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

- $\epsilon = 0.2$ typically
- Simple to implement, robust, parallelisable
- Replaced TRPO in most applications

---

## RLHF — Reinforcement Learning from Human Feedback

How ChatGPT was trained (Ouyang et al., 2022):

**Step 1 — Supervised Fine-Tuning (SFT)**:
Fine-tune GPT on human-written demonstrations.

**Step 2 — Reward Model Training**:
Humans rank model outputs. Train a reward model $r_\phi(x, y)$ to predict human preferences.

**Step 3 — PPO Fine-Tuning**:
$$\mathcal{L} = \mathbb{E}\left[r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{SFT}})\right]$$

The KL penalty prevents the model from deviating too far from the SFT model (reward hacking).

---

## AlphaGo / AlphaZero

**AlphaGo** (Silver et al., 2016): defeated Lee Sedol 4-1

Architecture:
- **Policy network**: $\pi(a \mid s)$ — which move to play
- **Value network**: $V(s)$ — who is winning
- **Monte Carlo Tree Search** (MCTS): guided by policy + value networks

**AlphaZero** (2017): trained from scratch with **self-play only** — no human games. Mastered Go, Chess, and Shogi.

> AlphaZero discovered novel strategies that human players had never considered in centuries of play.

---

## PyTorch: Simple Policy Gradient

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

def compute_returns(rewards, gamma=0.99):
    returns, R = [], 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    return (returns - returns.mean()) / (returns.std() + 1e-8)

# Training step
log_probs = torch.stack(log_probs)
returns   = compute_returns(rewards)
loss      = -(log_probs * returns).sum()
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

---

## Summary

| Algorithm | Type | Key Idea | Used in |
|-----------|------|---------|---------|
| Q-Learning | Value-based | TD learning | Tabular RL |
| DQN | Value-based | Neural Q + replay | Atari |
| REINFORCE | Policy gradient | Monte Carlo | Simple tasks |
| Actor-Critic | Hybrid | Advantage estimation | Continuous control |
| PPO | Policy gradient | Clipped objective | ChatGPT, robotics |
| AlphaZero | Model-based | Self-play + MCTS | Games |

**Next**: Generative Models — VAEs, GANs, and Diffusion.

Note: Lab: train DQN on CartPole-v1 using gymnasium. Observe the effect of removing experience replay or target network.
