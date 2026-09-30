# FreshGuard AI — Mathematical Formulations & Algorithm Derivations

## 1. Perceptual Difference Hashing (dHash) & Group Union-Find Algorithm

### Mathematical Definition of dHash
Given an RGB image $I$, resize to grayscale $I'$ of resolution $(W+1) \times H$ where $W=8, H=8$:
$$\text{bit}(x, y) = \begin{cases} 1 & \text{if } I'(x, y) > I'(x+1, y) \\ 0 & \text{otherwise} \end{cases}$$

The 64-bit integer hash $H(I)$ is constructed as:
$$H(I) = \sum_{k=0}^{63} \text{bit}_k \cdot 2^k$$

### Hamming Distance & Graph Grouping
The perceptual distance between two image hashes $H(I_a)$ and $H(I_b)$ is computed via bitwise XOR population count:
$$D(I_a, I_b) = \text{popcount}(H(I_a) \oplus H(I_b))$$

An undirected edge $(a, b)$ is created if $D(I_a, I_b) \le 8$. Connected components are computed using Disjoint Set Union (Union-Find) with path compression:
$$\text{Find}(x) = \begin{cases} x & \text{if } P[x] = x \\ \text{Find}(P[x]) & \text{otherwise} \end{cases}$$

## 2. Forward Pass Derivation

For an input image tensor $X \in \mathbb{R}^{B \times 3 \times 224 \times 224}$:

1. **Feature Extraction**:
   $$F = f_{\text{backbone}}(X) \in \mathbb{R}^{B \times C_{\text{feat}}}$$
   where $C_{\text{feat}} = 1280$ (EfficientNetV2-S), $768$ (ConvNeXt-Tiny), or $640$ (MobileViT).

2. **Standardized Classification Head**:
   $$\tilde{F} = \text{LayerNorm}(F) = \frac{F - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
   $$H = \text{GELU}(W_1 \tilde{F} + b_1), \quad W_1 \in \mathbb{R}^{256 \times C_{\text{feat}}}$$
   $$\hat{H} = \text{Dropout}(H, p=0.3)$$
   $$Z = W_2 \hat{H} + b_2 \in \mathbb{R}^{B \times 6}$$

3. **Softmax Probability & Abstention Logic**:
   $$P(Y = c \mid X) = \frac{e^{Z_c}}{\sum_{j=1}^6 e^{Z_j}}$$
   $$\text{Verdict}(X) = \begin{cases} \arg\max_{c} P(Y=c \mid X) & \text{if } \max_c P(Y=c \mid X) \ge 0.60 \\ -1 \quad (\text{Abstain / UNCERTAIN}) & \text{otherwise} \end{cases}$$

## 3. Training Optimization & Loss Function

Cross-Entropy Loss over mini-batch $N$:
$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^N \log P(Y = y_i \mid X_i)$$

Gradient update using AdamW with weight decay $\lambda$:
$$\theta_{t+1} = \theta_t - \eta_t \cdot \left( \frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_t \right)$$
where $\eta_t$ follows Cosine Annealing learning rate decay:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
