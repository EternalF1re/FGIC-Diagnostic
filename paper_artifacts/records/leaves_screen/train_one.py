"""Train one formal Phase 2B-S1 variant/fold under the shared protocol."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sklearn
import timm
import torch
import torchvision
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import Tensor, nn

from screen_core import (
    CONFIG_PATH,
    HERE,
    VARIANTS,
    ScreenModel,
    batch_mix,
    build_loaders,
    fold_class_weights,
    initialize_head,
    load_config,
    mixed_loss,
    model_counts,
    seed_everything,
    sha256_file,
)


def environment(device: torch.device) -> dict:
    result = {
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'timm': timm.__version__,
        'numpy': np.__version__,
        'sklearn': sklearn.__version__,
        'device': str(device),
        'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
    }
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        result.update({'gpu_index': index, 'gpu_name': props.name, 'gpu_memory_bytes': props.total_memory})
    return result


def evaluate(model: ScreenModel, loader, device: torch.device, collect: bool = False):
    model.eval()
    logits_parts: List[np.ndarray] = []
    feature_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    names: List[str] = []
    correct = total = 0
    with torch.inference_mode():
        for images, labels, indices, batch_names in loader:
            images = images.to(device, non_blocking=True)
            logits, features = model(images)
            predictions = logits.argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += len(labels)
            if collect:
                logits_parts.append(logits.float().cpu().numpy())
                feature_parts.append(features.float().cpu().numpy())
                label_parts.append(labels.numpy())
                index_parts.append(indices.numpy())
                names.extend(str(name) for name in batch_names)
    accuracy = correct / total
    if not collect:
        return accuracy
    labels_np = np.concatenate(label_parts).astype(np.int64)
    indices_np = np.concatenate(index_parts).astype(np.int64)
    logits_np = np.concatenate(logits_parts).astype(np.float32)
    features_np = np.concatenate(feature_parts).astype(np.float32)
    predictions_np = logits_np.argmax(axis=1).astype(np.int64)
    return {
        'accuracy': float(accuracy_score(labels_np, predictions_np)),
        'macro_f1': float(f1_score(labels_np, predictions_np, average='macro')),
        'balanced_accuracy': float(balanced_accuracy_score(labels_np, predictions_np)),
        'logits': logits_np,
        'features': features_np,
        'labels': labels_np,
        'sample_ids': indices_np,
        'image_names': np.asarray(names),
        'predictions': predictions_np,
    }


def checkpoint_payload(model: ScreenModel, variant: str, fold: int, stage: int, epoch: int, val_accuracy: float, config: dict) -> dict:
    return {
        'model_state_dict': model.state_dict(),
        'variant': variant,
        'fold': fold,
        'stage': stage,
        'epoch': epoch,
        'metric': val_accuracy,
        'metric_type': 'heldout_original_view_accuracy',
        'split_random_state': int(config['split']['split_random_state']),
        'training_seed': int(config['seed']['training_seed']),
        'config_sha256': sha256_file(CONFIG_PATH),
    }


def write_history(path: Path, rows: List[dict]) -> None:
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


def stage1_train(model, train_loader, val_loader, class_weights, device, config, variant, fold, output, history):
    protocol = config['common_training_protocol']
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    base_lrs = [float(protocol['stage1_backbone_lr']), float(protocol['stage1_head_lr'])]
    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': base_lrs[0]},
            {'params': head_params, 'lr': base_lrs[1]},
        ],
        betas=tuple(protocol['adam_betas']), eps=float(protocol['adam_eps']),
        weight_decay=float(protocol['stage1_weight_decay']),
    )
    epochs = int(protocol['stage1_epochs'])
    warmup = int(protocol['stage1_warmup_epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=float(protocol['stage1_label_smoothing'])
    )
    scaler = torch.amp.GradScaler(
        device.type, init_scale=float(protocol['amp_grad_scaler_init_scale']),
        enabled=device.type == 'cuda' and bool(protocol['amp'])
    )
    best_accuracy = -1.0
    best_epoch = -1
    checkpoint = output / 'best_stage1.pth'
    for epoch in range(epochs):
        model.train()
        amp_overflow_batches = 0
        if epoch < warmup:
            factor = (epoch + 1) / warmup
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group['lr'] = base_lr * factor
        started = time.perf_counter()
        losses = []
        for batch_index, (images, labels, _, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images, label_a, label_b, lam, _ = batch_mix(images, labels, epoch, batch_index, epochs, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda' and bool(protocol['amp'])):
                logits, _ = model(images)
                loss = mixed_loss(criterion, logits, label_a, label_b, lam)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite Stage1 loss at epoch {epoch}, batch {batch_index}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(protocol['gradient_clip_norm']))
            if not torch.isfinite(grad_norm):
                if not scaler.is_enabled():
                    raise FloatingPointError(f'Non-finite Stage1 FP32 gradient at epoch {epoch}, batch {batch_index}')
                scaler.step(optimizer)
                scaler.update()
                amp_overflow_batches += 1
                continue
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        val_accuracy = float(evaluate(model, val_loader, device))
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save(checkpoint_payload(model, variant, fold, 1, epoch, val_accuracy, config), checkpoint)
        history.append({
            'stage': 1, 'epoch': epoch, 'train_loss': float(np.mean(losses)),
            'val_accuracy': val_accuracy, 'best_val_accuracy': best_accuracy,
            'backbone_lr': optimizer.param_groups[0]['lr'], 'head_lr': optimizer.param_groups[1]['lr'],
            'amp_overflow_batches': amp_overflow_batches,
            'elapsed_seconds': time.perf_counter() - started,
        })
        write_history(output / 'training_history.csv', history)
        print(json.dumps({'variant': variant, 'fold': fold, 'stage': 1, 'epoch': epoch + 1, 'epochs': epochs, 'loss': history[-1]['train_loss'], 'val_accuracy': val_accuracy, 'best': best_accuracy}), flush=True)
        if epoch >= warmup:
            scheduler.step()
    return checkpoint, best_epoch, best_accuracy


def adapt_batch_norm(model, train_loader, device, batches: int) -> int:
    model.train()
    completed = 0
    with torch.no_grad():
        for images, _, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            model(images)
            completed += 1
            if completed >= batches:
                break
    return completed


def stage2_train(model, train_loader, val_loader, class_weights, device, config, variant, fold, output, history):
    protocol = config['common_training_protocol']
    epochs = int(protocol['stage2_epochs'])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(protocol['stage2_all_parameter_lr']),
        betas=tuple(protocol['adam_betas']), eps=float(protocol['adam_eps']),
        weight_decay=float(protocol['stage2_weight_decay']),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=float(protocol['stage2_label_smoothing'])
    )
    scaler = torch.amp.GradScaler(
        device.type, init_scale=float(protocol['amp_grad_scaler_init_scale']),
        enabled=device.type == 'cuda' and bool(protocol['amp'])
    )
    best_accuracy = -1.0
    best_epoch = -1
    checkpoint = output / 'best_stage2.pth'
    for epoch in range(epochs):
        model.train()
        amp_overflow_batches = 0
        started = time.perf_counter()
        losses = []
        for batch_index, (images, labels, _, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda' and bool(protocol['amp'])):
                logits, _ = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite Stage2 loss at epoch {epoch}, batch {batch_index}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(protocol['gradient_clip_norm']))
            if not torch.isfinite(grad_norm):
                if not scaler.is_enabled():
                    raise FloatingPointError(f'Non-finite Stage2 FP32 gradient at epoch {epoch}, batch {batch_index}')
                scaler.step(optimizer)
                scaler.update()
                amp_overflow_batches += 1
                continue
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        val_accuracy = float(evaluate(model, val_loader, device))
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save(checkpoint_payload(model, variant, fold, 2, epoch, val_accuracy, config), checkpoint)
        history.append({
            'stage': 2, 'epoch': epoch, 'train_loss': float(np.mean(losses)),
            'val_accuracy': val_accuracy, 'best_val_accuracy': best_accuracy,
            'backbone_lr': optimizer.param_groups[0]['lr'], 'head_lr': optimizer.param_groups[0]['lr'],
            'amp_overflow_batches': amp_overflow_batches,
            'elapsed_seconds': time.perf_counter() - started,
        })
        write_history(output / 'training_history.csv', history)
        print(json.dumps({'variant': variant, 'fold': fold, 'stage': 2, 'epoch': epoch + 1, 'epochs': epochs, 'loss': history[-1]['train_loss'], 'val_accuracy': val_accuracy, 'best': best_accuracy}), flush=True)
        scheduler.step()
    return checkpoint, best_epoch, best_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--fold', required=True, type=int, choices=range(5))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = load_config()
    variant, fold = args.variant, args.fold
    output = HERE / variant / f'fold_{fold}'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'run_manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing.get('status') == 'COMPLETE':
            print(json.dumps({'status': 'ALREADY_COMPLETE', 'variant': variant, 'fold': fold}))
            return
    device = torch.device(args.device)
    seed = int(config['seed']['training_seed'])
    seed_everything(seed)
    started = time.time()
    manifest = {
        'status': 'RUNNING', 'variant': variant, 'fold': fold, 'split_id': f'skf42_fold{fold}',
        'split_random_state': int(config['split']['split_random_state']), 'training_seed': seed,
        'device': str(device), 'started_unix': started, 'formal_training': True,
        'config_sha256': sha256_file(CONFIG_PATH),
        'train_one_sha256': sha256_file(Path(__file__).resolve()),
        'screen_core_sha256': sha256_file((HERE / 'screen_core.py').resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(CONFIG_PATH, output / 'config.json')
    try:
        train_loader, val_loader, dataset, train_idx, val_idx = build_loaders(fold, 1, config)
        np.savez_compressed(
            output / 'split_indices.npz', train_indices=train_idx, validation_indices=val_idx,
            split_random_state=np.asarray([config['split']['split_random_state']]),
            training_seed=np.asarray([seed]),
        )
        class_weights = fold_class_weights(dataset.labels, train_idx, int(config['dataset']['num_classes']))
        np.save(output / 'class_weights.npy', class_weights.numpy())
        model = ScreenModel(variant, int(config['dataset']['num_classes']), pretrained=True)
        initialize_head(model)
        counts = model_counts(model)
        model.to(device)
        history: List[dict] = []
        stage1_path, stage1_epoch, stage1_accuracy = stage1_train(
            model, train_loader, val_loader, class_weights, device, config, variant, fold, output, history
        )
        stage1_checkpoint = torch.load(stage1_path, map_location=device, weights_only=False)
        model.load_state_dict(stage1_checkpoint['model_state_dict'], strict=True)
        del train_loader, val_loader
        train_loader, val_loader, dataset, train_idx2, val_idx2 = build_loaders(fold, 2, config)
        if not np.array_equal(train_idx, train_idx2) or not np.array_equal(val_idx, val_idx2):
            raise ValueError('Stage1/Stage2 split mismatch')
        bn_batches = adapt_batch_norm(
            model, train_loader, device, int(config['common_training_protocol']['bn_adaptation_batches_before_stage2'])
        )
        stage2_path, stage2_epoch, stage2_accuracy = stage2_train(
            model, train_loader, val_loader, class_weights, device, config, variant, fold, output, history
        )
        stage2_checkpoint = torch.load(stage2_path, map_location=device, weights_only=False)
        model.load_state_dict(stage2_checkpoint['model_state_dict'], strict=True)
        final = evaluate(model, val_loader, device, collect=True)
        if not np.array_equal(final['sample_ids'], val_idx):
            raise ValueError('Final heldout order differs from stored validation indices')
        for key in ('logits', 'features', 'labels', 'sample_ids', 'image_names', 'predictions'):
            np.save(output / f'validation_{key}.npy', final[key])
        metrics = {key: final[key] for key in ('accuracy', 'macro_f1', 'balanced_accuracy')}
        metrics.update({
            'variant': variant, 'fold': fold, 'n': len(val_idx),
            'best_stage1_epoch_zero_based': stage1_epoch,
            'best_stage1_accuracy': stage1_accuracy,
            'best_stage2_epoch_zero_based': stage2_epoch,
            'best_stage2_accuracy': stage2_accuracy,
            'feature_extraction_location': config['architectures'][variant]['feature_extraction_location'],
        })
        (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest.update({
            'status': 'COMPLETE', 'completed_unix': time.time(), 'elapsed_seconds': time.time() - started,
            'environment': environment(device), 'model_counts': counts, 'bn_adaptation_batches_completed': bn_batches,
            'best_stage1_checkpoint': str(stage1_path.resolve()), 'best_stage1_sha256': sha256_file(stage1_path),
            'best_stage1_epoch_zero_based': stage1_epoch, 'best_stage1_accuracy': stage1_accuracy,
            'best_stage2_checkpoint': str(stage2_path.resolve()), 'best_stage2_sha256': sha256_file(stage2_path),
            'best_stage2_epoch_zero_based': stage2_epoch, 'best_stage2_accuracy': stage2_accuracy,
            'validation_artifacts': {key: f'validation_{key}.npy' for key in ('logits', 'features', 'labels', 'sample_ids', 'image_names', 'predictions')},
            'metrics': metrics,
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'status': 'COMPLETE', 'variant': variant, 'fold': fold, 'metrics': metrics}, ensure_ascii=False), flush=True)
    except Exception as exc:
        manifest.update({'status': 'FAILED', 'completed_unix': time.time(), 'elapsed_seconds': time.time() - started, 'error': repr(exc), 'traceback': traceback.format_exc()})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
