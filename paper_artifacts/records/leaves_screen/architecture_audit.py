"""Pre-training architecture and one-batch gradient audit for #0/#1/#2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from screen_core import (
    CONFIG_PATH,
    HERE,
    VARIANTS,
    ScreenModel,
    build_loaders,
    initialize_head,
    load_config,
    model_counts,
    seed_everything,
    sha256_file,
)


def parameter_table(model: nn.Module) -> str:
    lines = ['name\tshape\tparameters\trequires_grad']
    for name, parameter in model.named_parameters():
        lines.append(f'{name}\t{tuple(parameter.shape)}\t{parameter.numel()}\t{parameter.requires_grad}')
    return '\n'.join(lines) + '\n'


def audit_variant(variant: str, device: torch.device, pretrained: bool) -> dict:
    config = load_config()
    seed = int(config['seed']['training_seed'])
    seed_everything(seed)
    output = HERE / 'architecture_audit' / variant
    output.mkdir(parents=True, exist_ok=True)
    train_loader, _, _, _, _ = build_loaders(0, 99, config)
    images, labels, sample_ids, names = next(iter(train_loader))
    model = ScreenModel(variant, int(config['dataset']['num_classes']), pretrained=pretrained)
    initialize_head(model)
    model.to(device).train()
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    criterion = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    logits, features, trace = model(images, return_trace=True)
    loss = criterion(logits, labels)
    loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    all_gradients_present = all(gradient is not None for gradient in gradients)
    gradients_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None)
    result = {
        'variant': variant,
        'formal_training': False,
        'separate_from_formal_seed_stream': True,
        'device': str(device),
        'pretrained_backbone': pretrained,
        'config_path': str(CONFIG_PATH.resolve()),
        'config_sha256': sha256_file(CONFIG_PATH),
        'batch_size': int(images.shape[0]),
        'sample_ids_first_batch': sample_ids.tolist(),
        'names_first_five': list(names[:5]),
        'output_shape': list(logits.shape),
        'feature_shape': list(features.shape),
        'expected_output_shape': [int(images.shape[0]), int(config['dataset']['num_classes'])],
        'loss': float(loss.detach()),
        'loss_finite': bool(torch.isfinite(loss)),
        'all_trainable_gradients_present': all_gradients_present,
        'all_present_gradients_finite': gradients_finite,
        'forward_backward_elapsed_seconds': elapsed,
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'tensor_graph': {name: list(shape) for name, shape in trace.items()},
        'counts': model_counts(model),
        'mhsa_module_count': sum(isinstance(module, nn.MultiheadAttention) for module in model.modules()),
        'has_terminal_global_identity_late_addition': False if variant in ('#1', '#2') else None,
        'aggregation': 'mean of five stage tensors' if variant in ('#1', '#2') else 'not applicable',
        'shortcut_lambda': getattr(model.head, 'shortcut_lambda', None),
        'passed': bool(
            list(logits.shape) == [int(images.shape[0]), int(config['dataset']['num_classes'])]
            and torch.isfinite(loss)
            and all_gradients_present
            and gradients_finite
        ),
    }
    (output / 'module_tree.txt').write_text(str(model) + '\n', encoding='utf-8')
    (output / 'parameter_table.tsv').write_text(parameter_table(model), encoding='utf-8')
    (output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    del model, train_loader, images, labels, logits, features, loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    device = torch.device(args.device)
    results = [audit_variant(variant, device, not args.no_pretrained) for variant in VARIANTS]
    one = results[1]
    two = results[2]
    if one['tensor_graph'] != two['tensor_graph'] or one['counts'] != two['counts']:
        raise RuntimeError('#1/#2 differ in graph or parameter counts')
    summary = {
        'variants': list(VARIANTS),
        'all_passed': all(result['passed'] for result in results),
        'one_vs_two_same_tensor_graph': one['tensor_graph'] == two['tensor_graph'],
        'one_vs_two_same_parameter_counts': one['counts'] == two['counts'],
        'one_vs_two_only_declared_forward_change': 'fixed shortcut lambda 1.0 -> 0.1',
        'formal_training_started': False,
    }
    output = HERE / 'architecture_audit'
    (output / 'audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary['all_passed']:
        raise RuntimeError('Architecture smoke audit failed')


if __name__ == '__main__':
    main()
