from __future__ import annotations

import hashlib
from typing import Any

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


FEATURE_DIM = 2048


def create_backbone(pretrained: bool = True, model_name: str = "resnet50", expected_features: int | None = None) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    if expected_features is not None and int(model.num_features) != expected_features:
        raise ValueError(f"Expected a {expected_features}-D backbone, got {model.num_features}")
    return model


def initialize_linear(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def state_digest(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class ConventionalHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = F.dropout(F.silu(self.bn1(self.fc1(x))), 0.2, self.training)
        feature = F.dropout(F.silu(self.bn2(self.fc2(x))), 0.2, self.training)
        return self.classifier(feature), feature


class MappingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(256, 256)
        self.bn = nn.BatchNorm1d(256)

    def forward(self, x: Tensor) -> Tensor:
        mapped = F.dropout(F.silu(self.bn(self.linear(x))), 0.3, self.training)
        return mapped + 0.7 * x


class ProgressiveHead(nn.Module):
    """Final five-block head: fixed 0.7 shortcuts and mean aggregation."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_features, 256)
        self.mapping_blocks = nn.ModuleList([MappingBlock() for _ in range(5)])
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: Tensor, return_stages: bool = False):
        current = self.input_projection(x)
        stages = []
        for block in self.mapping_blocks:
            current = block(current)
            stages.append(current)
        stacked = torch.stack(stages, dim=1)
        feature = F.dropout(stacked.mean(dim=1), 0.2, self.training)
        output = (self.classifier(feature), feature)
        return (*output, stacked) if return_stages else output


class SingleBranchModel(nn.Module):
    def __init__(self, method: str, num_classes: int, pretrained: bool = True, backbone_name: str = "resnet50") -> None:
        super().__init__()
        if method not in {"ours_ft", "progressive"}:
            raise ValueError(method)
        self.method = method
        self.backbone = create_backbone(pretrained, backbone_name)
        feature_dim = int(self.backbone.num_features)
        self.head = ConventionalHead(feature_dim, num_classes) if method == "ours_ft" else ProgressiveHead(feature_dim, num_classes)
        initialize_linear(self.head)

    def forward(self, images: Tensor, return_stages: bool = False):
        features = self.backbone(images)
        if self.method == "progressive" and return_stages:
            return self.head(features, return_stages=True)
        return self.head(features)


class FeatureWiseGate(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, reduction: int = 16) -> None:
        super().__init__()
        hidden = feature_dim // reduction
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim, bias=False), nn.Sigmoid(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.layers(feature)


class DFAGModel(nn.Module):
    """Frozen anchor plus trainable plastic branch with feature-wise gating."""

    def __init__(self, num_classes: int, backbone_name: str = "resnet50") -> None:
        super().__init__()
        self.anchor = SingleBranchModel("ours_ft", num_classes, pretrained=False, backbone_name=backbone_name)
        self.plastic = SingleBranchModel("ours_ft", num_classes, pretrained=False, backbone_name=backbone_name)
        self.gate = FeatureWiseGate(int(self.plastic.backbone.num_features), reduction=16)
        self.freeze_anchor()

    def freeze_anchor(self) -> None:
        self.anchor.eval()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def load_stage1(self, payload: dict[str, Any], dataset: str, fold: int) -> None:
        if payload.get("method") != "ours_ft" or payload.get("dataset") != dataset or int(payload.get("fold", -1)) != fold:
            raise ValueError("DFAG Stage-1 provenance mismatch")
        state = payload["model_state_dict"]
        self.anchor.load_state_dict(state, strict=True)
        self.plastic.load_state_dict(state, strict=True)
        self.freeze_anchor()
        if state_digest(self.anchor.state_dict()) != state_digest(self.plastic.state_dict()):
            raise RuntimeError("Anchor/plastic initialization mismatch")

    def forward(self, images: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        with torch.no_grad():
            f_anc = self.anchor.backbone(images).detach()
        f_spec = self.plastic.backbone(images)
        gate = self.gate(f_spec)
        fused = gate * f_anc + (1.0 - gate) * f_spec
        logits, head_feature = self.plastic.head(fused)
        return logits, {"f_anc": f_anc, "f_spec": f_spec, "gate": gate, "fused": fused, "head_feature": head_feature}


def build_model(method: str, dataset: str, num_classes: int, pretrained: bool = True, backbone_name: str = "resnet50") -> nn.Module:
    if method in {"ours_ft", "progressive"}:
        return SingleBranchModel(method, num_classes, pretrained=pretrained, backbone_name=backbone_name)
    if method == "dfag":
        if pretrained:
            raise ValueError("DFAG must be initialized with --stage1-checkpoint")
        return DFAGModel(num_classes, backbone_name=backbone_name)
    if method == "l2_sp":
        from .baselines.l2_sp import L2SPModel
        return L2SPModel(num_classes, pretrained=pretrained)
    if method == "mc_loss":
        from .baselines.mc_loss import MCLossModel
        return MCLossModel(dataset, pretrained=pretrained)
    if method == "cal":
        from .baselines.cal import CALModel
        return CALModel(num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown method: {method}")
