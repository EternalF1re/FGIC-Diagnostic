import torch

from fgic_diagnostic.engine import batch_mix
from fgic_diagnostic.models import ConventionalHead, FeatureWiseGate, MappingBlock, ProgressiveHead


def test_progressive_head_shapes_and_five_stages():
    head = ProgressiveHead(2048, 200).eval()
    logits, feature, stages = head(torch.randn(3, 2048), return_stages=True)
    assert logits.shape == (3, 200)
    assert feature.shape == (3, 256)
    assert stages.shape == (3, 5, 256)


def test_mapping_block_has_fixed_residual_when_mapping_is_zero():
    block = MappingBlock().eval()
    with torch.no_grad():
        block.linear.weight.zero_()
        block.linear.bias.zero_()
    value = torch.randn(4, 256)
    assert torch.allclose(block(value), 0.7 * value)


def test_conventional_head_shape():
    head = ConventionalHead(2048, 17).eval()
    logits, feature = head(torch.randn(2, 2048))
    assert logits.shape == (2, 17)
    assert feature.shape == (2, 1024)


def test_gate_is_feature_wise_probability():
    gate = FeatureWiseGate().eval()(torch.randn(2, 2048))
    assert gate.shape == (2, 2048)
    assert torch.all((gate >= 0) & (gate <= 1))


def test_batch_mix_is_deterministic_for_epoch_and_batch():
    images = torch.randn(4, 3, 16, 16)
    labels = torch.arange(4)
    first = batch_mix(images, labels, 25, 3, 150)
    second = batch_mix(images, labels, 25, 3, 150)
    assert first[4:] == second[4:]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[2], second[2])

