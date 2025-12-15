# deep_learning_project
idea 3 (Gesture Classification Description)

Here’s a concise, publication‑ready section that covers exactly what’s needed: the comparison, pros/cons, and why one model (ViT) performs best on this task and dataset.

Comparison of the four models (our experimental results)
- Protocol: subject‑independent split (Train: 00–06, Val: 07–08, Test: 09 held‑out). Same preprocessing/augmentations and evaluation pipeline across models.
- Results (best runs):

| Model | Training regime | Best Val Acc | Test Acc (Subj 09) | Notes |
|---|---|---:|---:|---|
| ViT-B/16 | Transfer learning, two‑stage FT | 94.28% | 100.00% | Highest validation/generalization |
| Inception v1 (GoogLeNet) | Transfer learning + aux-loss | 89.17% | 99.85% | Strong, multi‑scale features |
| ResNet‑50 | Transfer learning | 88.50% | 99.95% | Stable, slightly below Inception on Val |
| VGG‑19 (from scratch) | Trained from scratch | ~77.4% | 98.65% | Heaviest and no pretraining |

- Additional observations:
  - All transfer‑learned models (ViT/ResNet/Inception) reached near‑perfect performance on the held‑out subject; ViT achieved perfect Test and AUC=1.00. VGG, despite lower Val, still scored very high on the (easier) held‑out subject.
  - Validation scores separate the models more clearly (ViT > Inception ≈ ResNet ≫ VGG‑scratch), which better reflects cross‑subject generalization.
  - Confusion matrices show most residual errors occur among visually similar gestures (e.g., index vs thumb, l vs ok), consistent across models but least frequent in ViT.

Pros and cons of each architecture
- ViT (Vision Transformer)
  - Pros:
    - Global self‑attention captures long‑range relations between fingers/shapes; robust to small rotations and crop variations.
    - Fine‑tuning converges quickly with two‑stage training (linear probe → full FT).
    - Achieved the highest Val/Test accuracy on our subject‑independent split.
  - Cons:
    - Heavier than Inception/ResNet; slower inference on modest GPUs.
    - Benefits from careful regularization and schedule; more sensitive to hyperparameters than ResNet.

- ResNet‑50
  - Pros:
    - Strong baseline; residual connections ease optimization and stabilize training.
    - Good speed/accuracy trade‑off; widely available, robust implementations.
  - Cons:
    - Primarily local receptive fields; can struggle more than ViT on subtle global hand configurations.
    - Slightly below Inception on our Val in this task.

- Inception v1 (GoogLeNet)
  - Pros:
    - Multi‑branch, multi‑scale filters (1×1/3×3/5×5) fit hand‑shape diversity well.
    - Lightweight (parameters) and fast at inference; auxiliary heads improve optimization.
    - Outperformed ResNet‑50 on Val in our runs.
  - Cons:
    - Training setup is a bit more complex (auxiliary losses and heads).
    - Still trails ViT on cross‑subject validation.

- VGG‑19 (from scratch)
  - Pros:
    - Simple, clean architecture; easy to implement from first principles (meets “from scratch” requirement).
  - Cons:
    - Very large parameter count without skip connections; needs much more data/epochs to generalize.
    - Lowest Val accuracy among the four under our compute/data budget.

Why ViT performs best for our task and dataset
- Dataset characteristics that favor ViT:
  - The gestures are defined by global finger configurations and relative spatial relationships. ViT’s self‑attention models these global dependencies directly, rather than relying on progressively enlarged local receptive fields.
  - Images are fairly controlled (centered hands, limited backgrounds), so ViT’s patch embedding plus light augmentations are enough to capture discriminative structures without heavy invariances.
- Training recipe that amplified ViT’s strengths:
  - Two‑stage fine‑tuning (head‑only → full FT with different LRs) stabilized adaptation and avoided catastrophic shifts in early epochs.
  - Grayscale + simplified augmentations (no horizontal flips/jitter) preserved class semantics (e.g., “L”) and reduced label‑flip noise.
  - Label smoothing + AMP helped regularize and speed training.
- Empirical evidence:
  - Highest Val accuracy (94.28%) on unseen subjects and perfect Test on the held‑out subject, with the cleanest confusion matrices and ROC curves.
  - Smaller error overlap on look‑alike gestures compared to CNN baselines.

Takeaway
- For LeapGestRecog under a subject‑independent split and our standardized pipeline, ViT offers the best cross‑subject generalization, Inception v1 slightly edges ResNet‑50 among CNNs thanks to multi‑scale processing, and VGG‑19 (from scratch) is the least competitive given its size and lack of pretraining.
