# P1 Stage1-only pooled OOF

Status: COMPLETE; all 15 checkpoint reproduction gates passed.

- Deterministic original-view inference only; model.eval() and torch.inference_mode().
- 18,353 unique samples per variant; exact Stage1/Stage2 ID and label alignment.
- FP32 logits; argmax equals saved prediction; first-batch repeated inference is bitwise identical.
- Paired bootstrap uses 100,000 replicates and seed 20260807; McNemar is exact two-sided.
- Fold directions describe five models from one controlled training seed, not independent-seed significance.
