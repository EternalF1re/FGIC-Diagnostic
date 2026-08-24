# Third-party notices

The maintained public package contains controlled PyTorch adaptations of the following methods. All three upstream code repositories identify their code license as MIT. Their original papers, authors and repositories must be cited when the corresponding baseline is used.

## L2-SP

- Xuhong Li, Yves Grandvalet and Franck Davoine, “Explicit Inductive Bias for Transfer Learning with Convolutional Networks,” ICML 2018.
- Upstream: https://github.com/holyseven/TransferLearningClassification
- Audited upstream commit: `1426356195d79613b12c94ad8d6d0db576b09261`
- Adaptation: identical inherited Conv/Linear-weight L2-SP reference penalty; newly initialized classifier receives ordinary L2.

## Mutual-Channel Loss

- Dongliang Chang et al., “The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification,” IEEE TIP 2020.
- Upstream: https://github.com/PRIS-CV/Mutual-Channel-Loss
- Audited upstream commit: `befb3692cd0d5382eb32fa4e093226247f609fd9`
- Adaptation: controlled pretrained ResNet-50 body with the official MC auxiliary-loss semantics and standard classifier inference.

## CAL

- Yongming Rao, Guangyi Chen, Jiwen Lu and Jie Zhou, “Counterfactual Attention Learning for Fine-Grained Visual Categorization and Re-identification,” ICCV 2021.
- Upstream: https://github.com/raoyongming/CAL
- Audited upstream commit: `0ba9d5084f2532eeb21c9ef051c23f8b339595ff`
- Adaptation: common pretrained ResNet-50, BAP/counterfactual objective, crop/drop augmentation, feature center, and formal two-rank SyncBN execution.

No dataset or pretrained-weight license is sublicensed by this repository.

