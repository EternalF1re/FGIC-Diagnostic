"""Complete CAL method body adapted to the common timm ResNet-50."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from common.core import create_backbone


EPSILON = 1e-6


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class BAP(nn.Module):
    def forward(self, features: Tensor, attentions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, channels, height, width = features.shape
        if attentions.shape[-2:] != (height, width):
            attentions = F.interpolate(attentions, size=(height, width), mode="bilinear", align_corners=False)
        matrix = torch.einsum("bmhw,bchw->bmc", attentions, features).div(float(height * width)).reshape(batch, -1)
        matrix = torch.sign(matrix) * torch.sqrt(torch.abs(matrix) + EPSILON)
        matrix = F.normalize(matrix, dim=-1)
        fake_attention = torch.empty_like(attentions).uniform_(0.0, 2.0) if self.training else torch.ones_like(attentions)
        counterfactual = torch.einsum("bmhw,bchw->bmc", fake_attention, features).div(float(height * width)).reshape(batch, -1)
        counterfactual = torch.sign(counterfactual) * torch.sqrt(torch.abs(counterfactual) + EPSILON)
        counterfactual = F.normalize(counterfactual, dim=-1)
        return matrix, counterfactual, fake_attention


def sample_attention_maps(attention_maps: Tensor) -> Tensor:
    selected = []
    for maps in attention_maps:
        weights = torch.sqrt(maps.sum(dim=(1, 2)).detach() + EPSILON)
        weights = F.normalize(weights, p=1, dim=0)
        indices = torch.multinomial(weights, 2, replacement=True)
        selected.append(maps[indices])
    return torch.stack(selected)


def batch_augment(images: Tensor, attention_map: Tensor, mode: str, theta: tuple[float, float], padding_ratio: float = 0.1) -> Tensor:
    batch, _, image_height, image_width = images.shape
    if mode == "crop":
        output = []
        for index in range(batch):
            current = attention_map[index:index + 1]
            threshold = random.uniform(*theta) * current.max()
            mask = F.interpolate(current, size=(image_height, image_width), mode="bilinear", align_corners=False) >= threshold
            nonzero = torch.nonzero(mask[0, 0], as_tuple=False)
            h_min = max(int(nonzero[:, 0].min().item() - padding_ratio * image_height), 0)
            h_max = min(int(nonzero[:, 0].max().item() + padding_ratio * image_height), image_height)
            w_min = max(int(nonzero[:, 1].min().item() - padding_ratio * image_width), 0)
            w_max = min(int(nonzero[:, 1].max().item() + padding_ratio * image_width), image_width)
            h_max = max(h_max, h_min + 1)
            w_max = max(w_max, w_min + 1)
            crop = images[index:index + 1, :, h_min:h_max, w_min:w_max]
            output.append(F.interpolate(crop, size=(image_height, image_width), mode="bilinear", align_corners=False))
        return torch.cat(output, dim=0)
    if mode == "drop":
        masks = []
        for index in range(batch):
            current = attention_map[index:index + 1]
            threshold = random.uniform(*theta) * current.max()
            masks.append(F.interpolate(current, size=(image_height, image_width), mode="bilinear", align_corners=False) < threshold)
        return images * torch.cat(masks, dim=0).float()
    raise ValueError(mode)


class CALModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_attentions = 32
        self.num_features = 2048
        self.backbone = create_backbone(pretrained)
        self.attentions = BasicConv2d(self.num_features, self.num_attentions)
        self.bap = BAP()
        self.classifier = nn.Linear(self.num_attentions * self.num_features, num_classes, bias=False)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        feature_maps = self.backbone.forward_features(images)
        attention_maps = self.attentions(feature_maps)
        feature_matrix, counterfactual_matrix, fake_attention = self.bap(feature_maps, attention_maps)
        raw_logits = self.classifier(feature_matrix * 100.0)
        counterfactual_logits = self.classifier(counterfactual_matrix * 100.0)
        causal_logits = raw_logits - counterfactual_logits
        selected_attention = sample_attention_maps(attention_maps) if self.training else attention_maps.mean(dim=1, keepdim=True)
        return {
            "raw_logits": raw_logits,
            "causal_logits": causal_logits,
            "counterfactual_logits": counterfactual_logits,
            "feature_matrix": feature_matrix,
            "attention_maps": attention_maps,
            "selected_attention": selected_attention,
            "fake_attention": fake_attention,
            "feature_maps": feature_maps,
        }


class CALFeatureCenter(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.register_buffer("value", torch.zeros(num_classes, 32 * 2048), persistent=True)
        self.update_count = 0

    def normalized_batch(self, labels: Tensor) -> Tensor:
        return F.normalize(self.value[labels], dim=-1)

    @torch.no_grad()
    def update(self, labels: Tensor, features: Tensor, beta: float = 0.05) -> None:
        normalized = self.normalized_batch(labels)
        self.value[labels] += beta * (features.detach() - normalized)
        self.update_count += 1


def training_objective(model: nn.Module, center: CALFeatureCenter, images: Tensor, labels: Tensor) -> dict[str, Tensor]:
    first = model(images)
    center_batch = center.normalized_batch(labels)
    center.update(labels, first["feature_matrix"], beta=0.05)
    with torch.no_grad():
        crop_images = batch_augment(images, first["selected_attention"][:, :1], "crop", (0.4, 0.6), padding_ratio=0.1)
        drop_images = batch_augment(images, first["selected_attention"][:, 1:], "drop", (0.2, 0.5))
    augmented_images = torch.cat([crop_images, drop_images], dim=0)
    augmented_labels = torch.cat([labels, labels], dim=0)
    second = model(augmented_images)
    all_aux = torch.cat([first["causal_logits"], second["causal_logits"]], dim=0)
    all_aux_labels = torch.cat([labels, augmented_labels], dim=0)
    center_loss = F.mse_loss(first["feature_matrix"], center_batch, reduction="sum") / images.shape[0]
    loss = (
        F.cross_entropy(first["raw_logits"], labels) / 3.0
        + F.cross_entropy(all_aux, all_aux_labels)
        + F.cross_entropy(second["raw_logits"], augmented_labels) * 2.0 / 3.0
        + center_loss
    )
    return {
        "loss": loss,
        "center_loss": center_loss,
        "primary_logits": first["causal_logits"],
        "raw_logits": first["raw_logits"],
        "attention_maps": first["attention_maps"],
        "fake_attention": first["fake_attention"],
        "crop_images": crop_images,
        "drop_images": drop_images,
        "second_forward_batch": torch.tensor(augmented_images.shape[0], device=images.device),
    }


def build_optimizer(model: CALModel) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)


def set_fractional_lr(optimizer: torch.optim.Optimizer, epoch: int, fraction: float) -> float:
    lr = 1e-3 * pow(0.9, (epoch + fraction) / 2.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr
