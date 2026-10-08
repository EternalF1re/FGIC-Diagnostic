# Final Cross-Backbone Protocol Manifest

- Config SHA256: `1a075d4241358e4681f2e7c70523a85bf286d97193fb1104b18bd713c5006fee`
- Dataset: Classify Leaves; 18,353 samples; 176 classes; five-fold StratifiedKFold with shuffle=True and random_state=42.
- Training seed: 42 for every fold; batch size: 64 for every backbone and configuration.
- Stage 1: 150 epochs; AdamW; backbone LR 0.0001; head LR 0.001; weight decay 0.001; 3-epoch warmup then cosine schedule.
- Stage 2: 60 epochs; all-parameter LR 5e-05; weight decay 0.0005; cosine schedule.
- Label smoothing: Stage 1 0.05; Stage 2 0.03.
- Stage-1 batch augmentation: CutMix p=0.5 alpha=1.0; Mixup p=0.4 alpha=0.4; none p=0.1; alpha linearly decays after 70% epochs
- Spatial augmentation both stages: Resize(299,299), RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), RandomRotation(180, fill white), ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), RandomAffine(degrees=0, shear=5), ToTensor, ImageNet normalization.
- BN recalibration: 50 no-grad training-mode batches before Stage-2 optimizer construction.
- AMP: FP16 with initial GradScaler scale 4096; gradient clipping 5.0.
- Checkpoint selection: maximum deterministic original-view held-out validation accuracy; exact tie selects earlier epoch; stage2 final checkpoint only from stage2
- Validation: Resize(299,299) -> ToTensor -> ImageNet normalization; original view; no TTA; no fold ensemble.
- DFAG Stage 1: exact selected Ours-FT Stage-1 checkpoint from the same new backbone/fold; copied to frozen anchor and plastic; DFAG trains Stage 2 only.
- Alpha 0.5: complete floating checkpoint state interpolation; nonfloating buffers copied from Stage 2; eval only; no BN refresh.
- Primary analysis: complete pooled five-fold OOF; 100,000 paired bootstrap resamples (seed 20260818); exact two-sided McNemar.

H=256 is fixed across backbones, so the projection compression ratio differs across architectures and is not independently controlled.
