# Controlled Reimplementation Augmentation Manifest

状态：`FROZEN_BEFORE_FORMAL_TRAINING`。所有 validation/OOF 均为 `Resize(299,299) -> ToTensor -> ImageNet Normalize`，单一原图路径；no TTA、no cross-fold ensemble。

| Method | Frozen 299 training transform | Source/adaptation status |
|---|---|---|
| L2-SP | Resize 342x342 (bilinear) -> ToTensor -> random brightness additive delta [-63/255,63/255] -> random saturation factor [0.5,1.5] -> random contrast factor [0.2,1.8] -> horizontal flip p=.5 -> random crop 299 -> ImageNet normalize | Official TF order is Resize256, the three random color ops, mirror, crop224. 342 is round(299*256/224). The official `blur` flag gates these color perturbations; it is not Gaussian blur. Each enabled color op is applied with a random magnitude, so application probability is 1. Dataset-mean subtraction is replaced by the locked timm ImageNet normalization required by the common pretrained artifact. No padding. |
| MC-Loss | Resize 299x299 (bilinear) -> RandomCrop299(padding=4) -> horizontal flip p=.5 -> ToTensor -> ImageNet normalize | Pretrained paper gives resize/input but not a complete executable augmentation (`NOT_SPECIFIED`). This predeclared minimal candidate uses the official from-scratch repository's resize/crop-padding/flip evidence, with only size and common-normalization adaptations. No color, vertical flip, or arbitrary rotation. |
| CAL | Resize341x341 (bilinear) -> RandomCrop299 -> horizontal flip p=.5 -> ColorJitter(brightness=.126,saturation=.5) -> ToTensor -> ImageNet normalize; plus method-internal attention crop theta(.4,.6), drop theta(.2,.5), and second forward | Direct official transform with `int(299/.875)=341`; method-internal crop/drop retained. Official eval crop/ensemble is not used under the common evaluation protocol. |
| Ours-FT / Progressive / DFAG | Resize299 -> horizontal flip p=.5 -> ColorJitter(.1 brightness/contrast/saturation/hue) -> RandomAffine(degrees=0,shear=5) -> ToTensor -> ImageNet normalize | Exact current audited CUB/Cars controlled transform. No vertical flip and no arbitrary rotation. Stage 2 retains this ordinary transform; Mixup/CutMix are disabled in Stage 2. |

No choice above was selected using smoke accuracy, held-out OOF, or official-test results.
