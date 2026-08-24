"""Standard L2-SP adaptation on the frozen common ResNet-50 backbone."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..models import create_backbone


class L2SPModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = create_backbone(pretrained, expected_features=2048)
        self.classifier = nn.Linear(2048, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        module_types = {name: type(module) for name, module in self.backbone.named_modules()}
        self._inherited_names: list[str] = []
        self._reference_buffers: dict[str, str] = {}
        for name, parameter in self.backbone.named_parameters():
            module_name, _, parameter_name = name.rpartition(".")
            if parameter_name == "weight" and module_types.get(module_name) in {nn.Conv2d, nn.Linear}:
                buffer_name = f"l2sp_reference_{len(self._inherited_names)}"
                self.register_buffer(buffer_name, parameter.detach().clone(), persistent=True)
                self._inherited_names.append(name)
                self._reference_buffers[name] = buffer_name

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.backbone(images))

    def regularization(self, alpha: float = 0.1, beta: float = 0.01) -> dict[str, Tensor]:
        parameters = dict(self.backbone.named_parameters())
        sp_raw = torch.zeros((), device=self.classifier.weight.device)
        for name in self._inherited_names:
            reference = getattr(self, self._reference_buffers[name])
            sp_raw = sp_raw + (parameters[name] - reference).square().sum()
        classifier_raw = self.classifier.weight.square().sum()
        return {
            "sp_raw": sp_raw,
            "classifier_raw": classifier_raw,
            "sp_term": 0.5 * alpha * sp_raw,
            "classifier_term": 0.5 * beta * classifier_raw,
        }

    def objective(self, images: Tensor, labels: Tensor, alpha: float = 0.1, beta: float = 0.01) -> dict[str, Tensor]:
        logits = self(images)
        ce = F.cross_entropy(logits, labels)
        regularization = self.regularization(alpha, beta)
        return {"loss": ce + regularization["sp_term"] + regularization["classifier_term"], "ce": ce, "logits": logits, **regularization}

    def parameter_partition(self) -> dict[str, Any]:
        return {
            "inherited_conv_or_linear_weights": list(self._inherited_names),
            "new_classifier_weight": "classifier.weight",
            "excluded": "all biases and all BatchNorm affine parameters",
            "optimizer_global_weight_decay": 0.0,
        }


def build_optimizer(model: L2SPModel) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)


def set_iteration_lr(optimizer: torch.optim.Optimizer, iteration: int) -> float:
    lr = 0.001 if iteration >= 6000 else 0.01
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr
