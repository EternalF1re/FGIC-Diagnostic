"""Compatibility entry point for the endpoint audit in the formal PyTorch env."""
from __future__ import annotations

import numpy as np
import torch

import supplementary_endpoint_audit as audit


def run_evaluate(model, loader, device):
    model.eval()
    labels_parts, ids_parts, logits_parts = [], [], []
    with torch.inference_mode():
        for images, labels, indices, _ in loader:
            logits, _ = model(images.to(device, non_blocking=True))
            logits_parts.append(logits.float().cpu().numpy())
            labels_parts.append(labels.numpy())
            ids_parts.append(indices.numpy())
    logits = np.concatenate(logits_parts).astype(np.float32)
    return {
        "sample_id": np.concatenate(ids_parts).astype(np.int64),
        "y_true": np.concatenate(labels_parts).astype(np.int64),
        "y_pred": logits.argmax(axis=1).astype(np.int64),
        "logits": logits,
    }


audit.evaluate = run_evaluate


if __name__ == "__main__":
    audit.main()
