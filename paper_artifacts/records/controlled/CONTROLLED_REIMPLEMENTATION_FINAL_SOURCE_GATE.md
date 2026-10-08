# Controlled Reimplementation Final Source and Table Gate

- `L2SP_SOURCE_TRACE = PASS`
- `MCLOSS_INFERENCE_FIDELITY = PASS`
- `FINAL_ADAPTATION_TABLE = FROZEN`
- frozen 60-job plan SHA: `1e4af9ba95615f58a08fe01371f9f0714ed4baf0e5768b5522ba1a28a5935e91` (PASS)

## L2-SP source trace

The paper objective separates inherited-weight SP (`alpha`) from ordinary L2 on new parameters (`beta`). Its Figure 1 evaluates a grid rather than declaring a universal optimum. The locked official implementation uses the standard `mode == 1` partition at `model/network_base.py:138-156`; the locked official Dogs command specifies `alpha=0.1`, `beta=0.01` at `run_classification/train.sh:5`. Therefore the correct record is: **source-backed values pre-specified before the controlled experiments and not tuned on CUB/Cars OOF results**.

The controlled implementation applies SP only to inherited Conv/Linear weights, excludes bias and BN affine, applies separate ordinary L2 to `classifier.weight`, and uses optimizer weight decay zero. It does not double-count generic WD plus SP.

## MC-Loss inference fidelity

The locked official implementation computes the MC supervisor only inside `if self.training` (`CUB-200-2011.py:154-172`); evaluation returns the standard classifier output. No channel-selection, CWA, special channel aggregation, multi-branch inference, or other method-specific test-time operation is required. The controlled implementation's evaluation path is standard GAP plus classifier logits. Training retains the exact CUB/Cars 2048-channel grouping and `CE + 0.005*(L_dis - 10*L_div)`.

## Frozen table

`CONTROLLED_REIMPLEMENTATION_FINAL_ADAPTATION_TABLE.csv` contains 12 dataset-specific rows (six methods x two datasets), including initialization, batch size, duration, optimizer, LR/schedule, momentum/WD, resolution, augmentation, smoothing, method coefficients, checkpoint rule, seed and inference protocol. The batch distinctions are explicit: L2-SP 64; MC-Loss 32; CAL global 16 (2 x 8); Ours-FT/Progressive/DFAG 32.
