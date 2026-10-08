"""Install the final dataset-specific transforms into the frozen smoke core."""

from __future__ import annotations

from typing import Any, Dict

from torchvision import transforms

import protocol_core


def train_transform(dataset_key: str) -> transforms.Compose:
    if dataset_key not in ("cub", "cars", "flowers"):
        raise ValueError(dataset_key)
    operations = [
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if dataset_key == "flowers":
        operations.append(transforms.RandomRotation(180, fill=(255, 255, 255)))
    operations.extend([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, shear=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(operations)


def serialized_transforms() -> Dict[str, Any]:
    result = {}
    for key in ("cub", "cars", "flowers"):
        train = train_transform(key)
        affine = next(operation for operation in train.transforms if isinstance(operation, transforms.RandomAffine))
        result[key] = {
            "train_repr": repr(train),
            "validation_repr": repr(protocol_core.eval_transform()),
            "official_test_repr": repr(protocol_core.eval_transform()),
            "random_rotation_present": any(isinstance(operation, transforms.RandomRotation) for operation in train.transforms),
            "vertical_flip_present": any(isinstance(operation, transforms.RandomVerticalFlip) for operation in train.transforms),
            "random_affine_degrees": list(affine.degrees),
            "random_affine_shear": list(affine.shear),
            "primary_tta": False,
        }
    return result


def install() -> None:
    protocol_core.train_transform = train_transform
    protocol_core.serialized_transforms = serialized_transforms

