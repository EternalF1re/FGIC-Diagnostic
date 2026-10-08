"""Remove case-insensitive duplicate keys from the final JSON manifest."""

from __future__ import annotations

import json
from pathlib import Path


path = Path(__file__).with_name("final_external_protocol_manifest.json")
record = json.loads(path.read_text(encoding="utf-8"))
lower_value = record.pop("new_config_sha256")
if lower_value != record["NEW_CONFIG_SHA256"]:
    raise ValueError("config SHA aliases disagree")
record["config_identity"] = {
    "path": record["new_config_path"],
    "sha256": lower_value,
}
path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "FIXED", "NEW_CONFIG_SHA256": record["NEW_CONFIG_SHA256"]}))
