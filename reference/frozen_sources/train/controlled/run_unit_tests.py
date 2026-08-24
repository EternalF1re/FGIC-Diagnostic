"""Fail-closed unit suite for the controlled reimplementation."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core import (  # noqa: E402
    DFAGModel,
    INPUT_SIZE,
    SingleBranchModel,
    canonical_sha,
    create_backbone,
    development_records,
    eval_transform,
    fold_manifest,
    locate_pretrained_artifact,
    pooled_oof_metrics,
    sha256_file,
    state_digest,
)
from methods.cal.model import BAP, CALFeatureCenter, training_objective  # noqa: E402
from methods.l2_sp.model import L2SPModel, build_optimizer as l2_optimizer  # noqa: E402
from methods.mc_loss.model import MCLossModel, cwa_mask, group_sizes, group_slices, mc_components  # noqa: E402


EXPECTED_ARTIFACT_SHA = "773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb"
EXPECTED_BACKBONE_SHA = "45179af731d5ea5470ba5906fadf0b55b884338ca64d2b7f53f46096665ec6c6"


def close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected} within {tolerance}")


def test_common() -> dict:
    manifests = {dataset: fold_manifest(dataset) for dataset in ("cub", "cars")}
    for dataset, expected_count in (("cub", 5994), ("cars", 8144)):
        manifest = manifests[dataset]
        assert manifest["development_count"] == expected_count
        ids = [row["sample_id"] for row in manifest["assignments"]]
        assert len(ids) == len(set(ids)) == expected_count
        counts = np.bincount([row["heldout_fold"] for row in manifest["assignments"]], minlength=5)
        assert int(counts.sum()) == expected_count and np.all(counts > 0)
        assert len(manifest["class_map"]) == (200 if dataset == "cub" else 196)
    record = development_records("cub")[0]
    with Image.open(record["path"]) as image:
        rgb = image.convert("RGB")
        first = eval_transform()(rgb)
        second = eval_transform()(rgb)
    assert tuple(first.shape) == (3, INPUT_SIZE, INPUT_SIZE)
    assert torch.equal(first, second)
    backbone = create_backbone(pretrained=True).eval()
    backbone_sha = state_digest(backbone.state_dict())
    assert backbone_sha == EXPECTED_BACKBONE_SHA
    with torch.inference_mode():
        features = backbone.forward_features(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
        pooled = backbone(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
    assert tuple(features.shape) == (1, 2048, 10, 10)
    assert tuple(pooled.shape) == (1, 2048)
    artifact = locate_pretrained_artifact()
    assert sha256_file(artifact) == EXPECTED_ARTIFACT_SHA
    rows = [{"sample_id": f"s{i}", "label": i % 3, "prediction": i % 3} for i in range(12)]
    metrics = pooled_oof_metrics(rows, [f"s{i}" for i in range(12)])
    assert metrics == {"accuracy": 1.0, "macro_f1": 1.0, "balanced_accuracy": 1.0}
    try:
        pooled_oof_metrics(rows[:-1], [f"s{i}" for i in range(12)])
        raise AssertionError("incomplete OOF was accepted")
    except ValueError:
        pass
    del backbone
    return {
        "fold_manifest_sha256": {key: value["fold_manifest_sha256"] for key, value in manifests.items()},
        "class_map_sha256": {key: value["class_map_sha256"] for key, value in manifests.items()},
        "artifact_sha256": EXPECTED_ARTIFACT_SHA,
        "backbone_state_sha256": backbone_sha,
        "feature_shape": [1, 2048, 10, 10],
        "eval_deterministic": True,
        "oof_semantics": "each expected sample exactly once; pooled metrics",
    }


def test_l2_sp() -> dict:
    torch.manual_seed(42)
    model = L2SPModel(7, pretrained=False)
    references_before = {name: getattr(model, buffer).clone() for name, buffer in model._reference_buffers.items()}
    regularization = model.regularization()
    assert regularization["sp_raw"].item() == 0.0
    regularization["sp_term"].backward()
    gradients = [dict(model.backbone.named_parameters())[name].grad for name in model._inherited_names]
    assert all(gradient is not None and torch.count_nonzero(gradient).item() == 0 for gradient in gradients)
    optimizer = l2_optimizer(model)
    assert all(group["weight_decay"] == 0.0 for group in optimizer.param_groups)
    partition = model.parameter_partition()
    assert partition["new_classifier_weight"] == "classifier.weight"
    assert all("bn" not in name.lower() for name in model._inherited_names)
    assert torch.allclose(regularization["classifier_raw"], model.classifier.weight.square().sum())
    for name, before in references_before.items():
        assert torch.equal(before, getattr(model, model._reference_buffers[name]))
        assert not getattr(model, model._reference_buffers[name]).requires_grad
    return {"inherited_weight_count": len(model._inherited_names), "sp_at_w0": 0.0, "sp_gradient_at_w0": 0.0, "optimizer_weight_decay": 0.0}


def test_mc_loss() -> dict:
    cub = group_sizes("cub")
    cars = group_sizes("cars")
    assert sum(cub) == sum(cars) == 2048
    assert cub[:152] == [10] * 152 and cub[152:] == [11] * 48
    assert cars[:108] == [10] * 108 and cars[108:] == [11] * 88
    assert group_slices(cub)[151] == slice(1510, 1520) and group_slices(cub)[152] == slice(1520, 1531)
    assert group_slices(cars)[107] == slice(1070, 1080) and group_slices(cars)[108] == slice(1080, 1091)
    first = cwa_mask(cub, torch.Generator().manual_seed(12345))
    second = cwa_mask(cub, torch.Generator().manual_seed(12345))
    assert torch.equal(first, second)
    for size, channel_slice in zip(cub, group_slices(cub)):
        values = first[0, channel_slice, 0, 0]
        assert int((values == 0).sum()) == size // 2
        assert int((values == 1).sum()) == size - size // 2
    features = torch.arange(2 * 2048 * 2 * 3, dtype=torch.float32).reshape(2, 2048, 2, 3) / 10000.0
    labels = torch.tensor([0, 199])
    parts = mc_components(features, labels, cub, mask=first)
    close(parts["l_dis"].item(), 5.349156379699707)
    close(parts["l_div"].item(), 0.09781821072101593)
    total = torch.tensor(1.234567) + 0.005 * (parts["l_dis"] - 10.0 * parts["l_div"])
    close(total.item(), 1.2564219236373901)
    same = torch.zeros(1, 2048, 1, 12)
    diverse = torch.zeros_like(same)
    same[:, :, :, 0] = 8.0
    for channel in range(2048):
        diverse[:, channel, :, channel % 12] = 8.0
    label = torch.tensor([0])
    same_div = mc_components(same, label, cub, mask=first)["l_div"]
    diverse_div = mc_components(diverse, label, cub, mask=first)["l_div"]
    assert diverse_div > same_div
    model = MCLossModel("cub", pretrained=False).eval()
    images = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        direct = model(images)
        expected = model.standard_logits(model.backbone.forward_features(images))
    assert torch.equal(direct, expected)
    return {
        "cub_total": sum(cub), "cars_total": sum(cars), "seeded_mask_reproducible": True,
        "golden_l_dis": parts["l_dis"].item(), "golden_l_div": parts["l_div"].item(),
        "golden_total_from_ce_1_234567": total.item(), "diversity_sign": "PASS",
        "dynamic_hw": list(features.shape[-2:]), "inference_mc_branch_disabled": True,
    }


class TinyCAL(torch.nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
        self.classes = classes
        self.calls: list[int] = []

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        self.calls.append(batch)
        base = images.mean(dim=(1, 2, 3), keepdim=False) * self.scale
        raw = base[:, None].repeat(1, self.classes)
        raw = raw + torch.arange(self.classes, device=images.device)[None] * self.scale
        causal = raw - 0.25 * raw
        feature = base[:, None].repeat(1, 32 * 2048)
        attention = torch.relu(images.mean(dim=1, keepdim=True)).repeat(1, 32, 1, 1) + 0.01
        selected = attention[:, :2] if self.training else attention.mean(dim=1, keepdim=True)
        fake = torch.rand_like(attention) * 2.0 if self.training else torch.ones_like(attention)
        return {"raw_logits": raw, "causal_logits": causal, "feature_matrix": feature, "attention_maps": attention, "selected_attention": selected, "fake_attention": fake}


def test_cal() -> dict:
    features = torch.randn(2, 2048, 3, 4)
    attention = torch.relu(torch.randn(2, 32, 3, 4))
    bap = BAP().train()
    matrix, counterfactual, fake = bap(features, attention)
    assert matrix.shape == counterfactual.shape == (2, 32 * 2048)
    assert fake.min() >= 0 and fake.max() <= 2
    bap.eval()
    _, _, fake_eval = bap(features, attention)
    assert torch.equal(fake_eval, torch.ones_like(fake_eval))
    model = TinyCAL(5).train()
    center = CALFeatureCenter(5)
    before = center.value.clone()
    images = torch.randn(2, 3, 16, 16)
    labels = torch.tensor([0, 1])
    output = training_objective(model, center, images, labels)
    assert model.calls == [2, 4]
    assert int(output["second_forward_batch"]) == 4
    assert center.update_count == 1 and not torch.equal(before, center.value)
    output["loss"].backward()
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
    model.eval()
    model.calls.clear()
    center_before_eval = center.value.clone()
    with torch.inference_mode():
        evaluation = model(images)
    assert model.calls == [2] and evaluation["selected_attention"].shape[1] == 1
    assert torch.equal(center_before_eval, center.value)
    return {
        "attention_shape": [2, 32, 3, 4], "bap_shape": [2, 65536],
        "fake_attention_train_uniform_0_2": True, "fake_attention_eval_ones": True,
        "causal_logits_present": True, "feature_center_train_only": True,
        "crop_drop_second_forward": True, "heldout_single_original_forward": True,
    }


def test_ours_progressive_dfag() -> dict:
    ours = SingleBranchModel("ours_ft", 7, pretrained=False)
    progressive = SingleBranchModel("progressive", 7, pretrained=False)
    assert len(progressive.head.mapping_blocks) == 5
    names = " ".join(name.lower() for name, _ in progressive.named_modules())
    assert "mhsa" not in names and "ms_lfa" not in names and "terminal" not in names
    assert not any("lambda" in name.lower() for name, _ in progressive.named_parameters())
    payload = {"method": "ours_ft", "dataset": "cub", "fold": 2, "model_state_dict": ours.state_dict()}
    dfag = DFAGModel(7)
    dfag.load_same_fold_stage1(payload, "cub", 2)
    assert state_digest(dfag.anchor.state_dict()) == state_digest(dfag.plastic.state_dict())
    assert all(not parameter.requires_grad for parameter in dfag.anchor.parameters())
    assert all(parameter.requires_grad for parameter in dfag.plastic.parameters())
    gate = dfag.gate(torch.randn(3, 2048))
    assert gate.shape == (3, 2048) and gate.min() >= 0 and gate.max() <= 1
    try:
        dfag.load_same_fold_stage1(payload, "cars", 2)
        raise AssertionError("DFAG accepted wrong-dataset Stage-1")
    except ValueError:
        pass
    return {
        "progressive_blocks": 5, "shortcut_lambda": 0.7, "learnable_lambda": False,
        "mhsa": False, "terminal_residual": False, "obsolete_ms_lfa_name": False,
        "dfag_same_fold_provenance": True, "anchor_frozen": True, "plastic_trainable": True,
        "feature_wise_gate_shape": [3, 2048],
    }


def main() -> int:
    output_json = ROOT / "preflight" / "unit_test_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tests = {
        "common": test_common,
        "l2_sp": test_l2_sp,
        "mc_loss": test_mc_loss,
        "cal": test_cal,
        "ours_progressive_dfag": test_ours_progressive_dfag,
    }
    results = {}
    for name, function in tests.items():
        try:
            results[name] = {"status": "PASS", "details": function()}
        except Exception as error:
            results[name] = {"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc()}
    payload = {
        "schema_version": 1,
        "suite": "controlled_reimplementation_unit_tests",
        "status": "PASS" if all(row["status"] == "PASS" for row in results.values()) else "FAIL",
        "tests": results,
        "result_sha256": canonical_sha(results),
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
