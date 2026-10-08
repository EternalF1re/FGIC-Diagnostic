"""Pretrained ResNet-50 Mutual-Channel Loss controlled implementation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from common.core import create_backbone


def group_sizes(dataset: str) -> list[int]:
    if dataset == "cub":
        sizes = [10] * 152 + [11] * 48
    elif dataset == "cars":
        sizes = [10] * 108 + [11] * 88
    else:
        raise ValueError(dataset)
    if sum(sizes) != 2048:
        raise AssertionError("MC channel groups must cover exactly 2048 channels")
    return sizes


def group_slices(sizes: list[int]) -> list[slice]:
    result = []
    start = 0
    for size in sizes:
        result.append(slice(start, start + size))
        start += size
    if start != 2048:
        raise AssertionError(start)
    return result


def cwa_mask(sizes: list[int], generator: torch.Generator | None = None, device: torch.device | None = None) -> Tensor:
    pieces = []
    for size in sizes:
        zeros = size // 2
        values = torch.cat([torch.zeros(zeros), torch.ones(size - zeros)])
        permutation = torch.randperm(size, generator=generator)
        pieces.append(values[permutation])
    return torch.cat(pieces).to(device=device).view(1, 2048, 1, 1)


def mc_components(
    features: Tensor,
    labels: Tensor,
    sizes: list[int],
    mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    if features.ndim != 4 or features.shape[1] != 2048:
        raise ValueError(f"expected [B,2048,H,W], got {tuple(features.shape)}")
    if mask is None:
        mask = cwa_mask(sizes, generator=generator, device=features.device)
    masked = features * mask
    spatial_attention = torch.softmax(features.flatten(2), dim=2)
    discrimination_scores = []
    diversity_scores = []
    for size, channel_slice in zip(sizes, group_slices(sizes)):
        discrimination_scores.append(masked[:, channel_slice].amax(dim=1).mean(dim=(1, 2)))
        per_location_max = spatial_attention[:, channel_slice].amax(dim=1)
        diversity_scores.append(per_location_max.sum(dim=1) / float(size))
    class_scores = torch.stack(discrimination_scores, dim=1)
    l_dis = F.cross_entropy(class_scores, labels)
    l_div = torch.stack(diversity_scores, dim=1).mean()
    return {"l_dis": l_dis, "l_div": l_div, "class_scores": class_scores, "mask": mask}


class MCLossModel(nn.Module):
    def __init__(self, dataset: str, pretrained: bool = True) -> None:
        super().__init__()
        self.dataset = dataset
        self.sizes = group_sizes(dataset)
        self.num_classes = len(self.sizes)
        self.backbone = create_backbone(pretrained)
        self.classifier = nn.Linear(2048, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def standard_logits(self, features: Tensor) -> Tensor:
        return self.classifier(features.mean(dim=(2, 3)))

    def forward(self, images: Tensor, labels: Tensor | None = None, generator: torch.Generator | None = None):
        features = self.backbone.forward_features(images)
        logits = self.standard_logits(features)
        if not self.training or labels is None:
            return logits
        ce = F.cross_entropy(logits, labels)
        components = mc_components(features, labels, self.sizes, generator=generator)
        loss = ce + 0.005 * (components["l_dis"] - 10.0 * components["l_div"])
        return {"loss": loss, "ce": ce, "logits": logits, "features": features, **components}


def build_optimizer(model: MCLossModel) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        [
            {"params": model.backbone.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-2},
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )


def set_epoch_lr(optimizer: torch.optim.Optimizer, epoch: int) -> tuple[float, float]:
    multiplier = 0.01 if epoch >= 225 else 0.1 if epoch >= 150 else 1.0
    rates = (1e-4 * multiplier, 1e-2 * multiplier)
    for group, lr in zip(optimizer.param_groups, rates):
        group["lr"] = lr
    return rates
