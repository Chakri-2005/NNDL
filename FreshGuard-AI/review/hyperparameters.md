# FreshGuard AI — Hyperparameter Matrix & Justifications

## 1. Central Hyperparameter Configuration

| Hyperparameter | Value | Scope | Scientific & Empirical Justification |
| :--- | :---: | :--- | :--- |
| **Input Resolution** | $224 \times 224$ | All Models | Standard ImageNet resolution balancing feature detail with fast edge inference. |
| **Batch Size** | $32$ | DataLoader | Optimal GPU memory utilization and stable mini-batch gradient estimation. |
| **Stage 1 Epochs** | $3$ | Warm-up | Allows the new random classification head to adapt without disrupting backbone weights. |
| **Stage 2 Epochs** | $5$ | Fine-Tuning | Sufficient for transfer learning convergence without risk of catastrophic forgetting. |
| **Stage 1 LR ($\eta_1$)** | $10^{-3}$ | Head | Standard initial learning rate for randomly initialized dense linear layers. |
| **Stage 2 LR ($\eta_2$)** | $10^{-4}$ | Backbone | Reduced learning rate prevents destroying pretrained ImageNet feature representations. |
| **Optimizer** | AdamW | Optimization | Decouples weight decay from gradient updates, improving generalization. |
| **Weight Decay ($\lambda$)** | $10^{-4}$ | Regularization | $L_2$ penalty preventing overfitting on minor color or background variations. |
| **Dropout Rate** | $0.3$ | Classifier Head | Prevents co-adaptation of features in the 256-dimensional hidden layer. |
| **Confidence Threshold** | $0.60$ | Inference API | Out-of-Distribution safety threshold ensuring model abstains on ambiguous inputs. |
| **Random Seed** | $42$ | Global | Guarantees exact reproducibility across dataset splits and weight initializations. |

## 2. Augmentation Pipeline Hyperparameters

| Augmentation Step | Parameter Value | Applied To | Rationale |
| :--- | :--- | :--- | :--- |
| **RandomHorizontalFlip** | $p = 0.5$ | Train Only | Fruit freshness visual characteristics are invariant to horizontal orientation. |
| **RandomRotation** | $\pm 15^\circ$ | Train Only | Simulates hand-held mobile camera tilt without excessive boundary cropping. |
| **ColorJitter** | Brightness=0.1, Contrast=0.1 | Train Only | Simulates real-world indoor/outdoor mobile lighting changes. |
| **Resize & Center Crop** | $224 \times 224$ | Val / Test | Deterministic preprocessing ensuring unbiased evaluation. |
| **ImageNet Normalization** | $\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$ | All Sets | Aligns input tensors with ImageNet pretrained feature distribution. |
