# Source provenance map

The maintained package is a path-independent extraction of the final audited implementations. The exact experiment-time files are retained under `reference/frozen_sources`.

| Maintained public source | Frozen source | Role |
|---|---|---|
| `src/fgic_diagnostic/models.py` | `reference/frozen_sources/src/common/core.py` | Ours-FT, Progressive Head and DFAG architecture |
| `src/fgic_diagnostic/baselines/l2_sp.py` | `reference/frozen_sources/src/methods/l2_sp/model.py` | Controlled L2-SP |
| `src/fgic_diagnostic/baselines/mc_loss.py` | `reference/frozen_sources/src/methods/mc_loss/model.py` | Controlled MC-Loss |
| `src/fgic_diagnostic/baselines/cal.py` | `reference/frozen_sources/src/methods/cal/model.py` | Controlled CAL body |
| `src/fgic_diagnostic/engine.py` | `reference/frozen_sources/train/controlled/formal_single_job.py` | Stage 1/2 training and held-out selection |
| `src/fgic_diagnostic/cal_ddp.py` | final controlled CAL DDP runner | Two-rank CAL topology |
| `src/fgic_diagnostic/metrics.py` | `reference/frozen_sources/evaluation` | pooled metrics and paired inference |
| `configs/controlled` | final controlled JSON configs | frozen CUB/Cars protocol |
| `reference/frozen_sources/train/cross_backbone` | final cross-backbone scripts | result-forensic snapshot |
| `reference/frozen_sources/diagnostics/dfag` | final Phase2F scripts | result-forensic DFAG diagnostics |

Only the maintained `fgic` and `fgic-cal-ddp` entry points are supported for new runs. No scientific constant was imported from a superseded result directory.

