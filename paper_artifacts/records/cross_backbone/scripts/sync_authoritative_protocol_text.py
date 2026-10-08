"""Mechanically synchronize authoritative protocol fields into the new config."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / "final_cross_backbone_protocol_config.json"
source = root.parent / "phase2b_screen" / "configs" / "controlled_screen.json"

new = json.loads(target.read_text(encoding="utf-8"))
authority = json.loads(source.read_text(encoding="utf-8"))
for key in (
    "checkpoint_selection",
    "stage1_batch_augmentation",
    "spatial_augmentation_both_stages",
):
    new["common_training_protocol"][key] = authority["common_training_protocol"][key]

target.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
