# P0 Phase2A Ours-FT versus Phase2B #0 protocol provenance

Verdict: **C. NOT DIRECTLY COMPARABLE**

Directly measured OOF accuracies: Phase2A 0.975371873808097 (97.5372%), Phase2B 0.977006483953577 (97.7006%), difference 0.1635 pp.

Field counts: SAME=19, DIFFERENT=4, UNRESOLVED=36.

The dataset, exact folds, architecture, and original-view OOF rule align. However, historical Phase2A training seeds, optimizer/LR/augmentation/class-weight/BN/AMP details are not serialized in the selected checkpoints, while checkpoint-candidate and selection mechanisms differ. These protocol differences and unresolved fields prevent direct attribution of the +0.1634 pp gap.

PHASE2A_BASELINE_PROVENANCE_NOT_FULLY_RECOVERED

| # | Field | Status | Phase2A | Phase2B |
|---:|---|---|---|---|
| 1 | dataset root | SAME | <LOCAL_PATH>| <LOCAL_PATH>|
| 2 | train.csv | SAME | .../classify-leaves/train.csv | .../classify-leaves/train.csv |
| 3 | label mapping | SAME | first appearance order in train.csv | first appearance order in train.csv |
| 4 | num classes | SAME | 176 | 176 |
| 5 | input resolution | SAME | 299x299 | 299x299 |
| 6 | split method | SAME | StratifiedKFold | StratifiedKFold |
| 7 | number of folds | SAME | 5 | 5 |
| 8 | shuffle | SAME | True | True |
| 9 | split_random_state | SAME | 42 | 42 |
| 10 | exact fold assignments | SAME | artifact-verified exact held-out membership | artifact-verified exact held-out membership |
| 11 | training seed | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 42 |
| 12 | seed is +fold | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | False; same seed every fold |
| 13 | DataLoader seed | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | controlled generator from seed 42 |
| 14 | worker seed | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | worker_init_fn seeded |
| 15 | model backbone | SAME | timm inception_resnet_v2; GAP 1536-D | timm inception_resnet_v2; GAP 1536-D |
| 16 | pretrained source | UNRESOLVED | checkpoint proves architecture, not initialization source | timm ImageNet pretrained |
| 17 | baseline head architecture | SAME | 1536-1024-BN-SiLU-Drop0.2-1024-BN-SiLU-Drop0.2-176 | same |
| 18 | feature extraction location | DIFFERENT | Phase2A Ours-FT run persists logits but no head feature; evaluator internal specific is 1536-D backbone GAP | saved 1024-D second head-block output after Dropout |
| 19 | batch size | UNRESOLVED | training batch unresolved; checkpoint-only inference used 32 | training 64 |
| 20 | num_workers | UNRESOLVED | training workers unresolved; checkpoint-only inference used 4 | training 4 |
| 21 | Stage1 epochs | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 150 |
| 22 | Stage1 optimizer | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | AdamW |
| 23 | backbone LR | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 1e-4 |
| 24 | head LR | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 1e-3 |
| 25 | weight decay | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 1e-3 |
| 26 | scheduler | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | linear warmup then cosine T_max=147 eta_min=1e-5 |
| 27 | warmup | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 3 epochs |
| 28 | label smoothing | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 0.05 |
| 29 | Mixup | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | p=0.4 alpha=0.4 |
| 30 | CutMix | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | p=0.5 alpha=1.0 |
| 31 | augmentation attenuation schedule | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | alpha linearly decays after 70% epochs |
| 32 | spatial augmentation | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | Resize, H/V flip, rotation, ColorJitter, affine, normalize |
| 33 | class weighting | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | inverse fold-train frequency, clipped/normalized, minority x1.1 |
| 34 | class weight based on full train.csv | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | False |
| 35 | class weight based on fold training subset | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | True |
| 36 | Stage1 checkpoint candidate mechanism | DIFFERENT | multiple candidate-index Stage2 artifacts (_0/_1/_2); exact Stage1 links not serialized | single best Stage1 checkpoint feeds one Stage2 run |
| 37 | ever used top-3 candidate | DIFFERENT | three candidate indices are preserved per fold; historical script linkage remains partial | No |
| 38 | current is single-best checkpoint | SAME | one highest-metric Stage2 checkpoint per fold used for Phase2A OOF | one best Stage2 checkpoint per fold used for Phase2B OOF |
| 39 | exact checkpoint selection rule | DIFFERENT | highest stored/filename val_acc among existing artifacts; lexical filename tie-break | max deterministic held-out accuracy; exact tie selects earlier epoch |
| 40 | Stage2 epochs | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 60 |
| 41 | Stage2 optimizer | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | AdamW |
| 42 | Stage2 LR | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 5e-5 all parameters |
| 43 | Stage2 weight decay | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 5e-4 |
| 44 | Stage2 scheduler | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | CosineAnnealingLR T_max=60 eta_min=1e-6 |
| 45 | Stage2 label smoothing | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 0.03 |
| 46 | Stage2 augmentation | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | spatial augmentation; no batch Mixup/CutMix |
| 47 | BN adaptation | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | Yes |
| 48 | 50-batch no-grad train-mode BN update | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | 50 batches |
| 49 | BN adaptation before/after optimizer creation | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | before Stage2 optimizer creation |
| 50 | gradient clipping | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | global norm 5.0 |
| 51 | gradient clipping semantics/historical bug | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | unscale before clipping in controlled runner |
| 52 | AMP | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | True, float16 |
| 53 | GradScaler | UNRESOLVED | UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources | enabled, init_scale=4096 |
| 54 | validation transform | SAME | Resize299, ToTensor, ImageNet Normalize | same |
| 55 | original view/TTA | SAME | original view only for the 97.5372 metric | original view only for 97.700648 metric |
| 56 | softmax averaging | SAME | none for original-view OOF | none for original-view OOF |
| 57 | fold ensemble | SAME | none; each sample predicted only by its held-out fold model | none; each sample predicted only by its held-out fold model |
| 58 | OOF assembly rule | SAME | concatenate five mutually exclusive held-out folds to 18,353 | same |
| 59 | final metric source | SAME | pooled argmax accuracy from Phase2A checkpoint-only logits = 0.975371873808097 | pooled accuracy from Phase2B saved Stage2 OOF predictions = 0.977006483953577 |
