"""Summarize completed Phase 2B-S1 OOF runs and enforce the stop gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from screen_core import HERE, VARIANTS, load_config


DISPLAY = {'#0': '#0 Original baseline', '#1': '#1 Deep/narrow λ=1.0', '#2': '#2 Scaling-only λ=0.1'}


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'macro_f1': float(f1_score(labels, predictions, average='macro')),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
    }


def load_variant(variant: str) -> pd.DataFrame:
    parts = []
    for fold in range(5):
        output = HERE / variant / f'fold_{fold}'
        manifest = json.loads((output / 'run_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status') != 'COMPLETE':
            raise RuntimeError(f'{variant} fold {fold} is not complete')
        sample_ids = np.load(output / 'validation_sample_ids.npy').astype(np.int64)
        labels = np.load(output / 'validation_labels.npy').astype(np.int64)
        logits = np.load(output / 'validation_logits.npy').astype(np.float32)
        predictions = np.load(output / 'validation_predictions.npy').astype(np.int64)
        names = np.load(output / 'validation_image_names.npy').astype(str)
        if not np.array_equal(logits.argmax(axis=1), predictions):
            raise ValueError(f'Logit/prediction mismatch: {variant} fold {fold}')
        parts.append(pd.DataFrame({
            'dataset_index': sample_ids, 'image': names, 'label': labels,
            'prediction': predictions, 'fold': fold,
        }))
    result = pd.concat(parts, ignore_index=True).sort_values('dataset_index').reset_index(drop=True)
    if not np.array_equal(result.dataset_index.to_numpy(), np.arange(18353)):
        raise ValueError(f'OOF coverage mismatch: {variant}')
    result['correct'] = result.prediction == result.label
    return result


def paired(a: pd.DataFrame, b: pd.DataFrame, name: str, fold: str | int, replicates: int, seed: int) -> dict:
    if not np.array_equal(a.dataset_index, b.dataset_index) or not np.array_equal(a.label, b.label):
        raise ValueError('Paired OOF alignment failure')
    ca = a.correct.to_numpy(bool)
    cb = b.correct.to_numpy(bool)
    n = len(a)
    n10 = int(np.sum(ca & ~cb))
    n01 = int(np.sum(~ca & cb))
    n11 = int(np.sum(ca & cb))
    n00 = int(np.sum(~ca & ~cb))
    rng = np.random.default_rng(seed + (int(fold) if fold != 'POOLED_OOF' else 100))
    draws = rng.multinomial(n, np.asarray([n10, n - n10 - n01, n01]) / n, size=replicates)
    deltas = (draws[:, 2] - draws[:, 0]) / n
    low, high = np.quantile(deltas, [0.025, 0.975])
    discordant = n10 + n01
    accuracy_a, accuracy_b = float(ca.mean()), float(cb.mean())
    delta_pp = 100.0 * (accuracy_b - accuracy_a)
    return {
        'comparison': name,
        'scope': fold if fold == 'POOLED_OOF' else f'fold_{fold}',
        'n': n,
        'accuracy_a': accuracy_a,
        'accuracy_b': accuracy_b,
        'delta_b_minus_a_pp': delta_pp,
        'direction': 'positive' if delta_pp > 0 else ('negative' if delta_pp < 0 else 'tie'),
        'prediction_changed_n': int(np.sum(a.prediction.to_numpy() != b.prediction.to_numpy())),
        'n11_both_correct': n11,
        'n10_a_correct_b_wrong': n10,
        'n01_a_wrong_b_correct': n01,
        'n00_both_wrong': n00,
        'correctness_discordant_n': discordant,
        'paired_bootstrap_95ci_low_pp': 100.0 * float(low),
        'paired_bootstrap_95ci_high_pp': 100.0 * float(high),
        'bootstrap_replicates': replicates,
        'bootstrap_seed': seed,
        'mcnemar_exact_p': float(binomtest(min(n10, n01), discordant, .5).pvalue) if discordant else 1.0,
    }


def judgement(row: pd.Series, fold_rows: pd.DataFrame, secondary_a: pd.Series, secondary_b: pd.Series) -> str:
    positive = int((fold_rows.delta_b_minus_a_pp > 0).sum())
    negative = int((fold_rows.delta_b_minus_a_pp < 0).sum())
    ties = int((fold_rows.delta_b_minus_a_pp == 0).sum())
    secondary_delta = 100.0 * float(secondary_b.macro_f1 - secondary_a.macro_f1)
    balanced_delta = 100.0 * float(secondary_b.balanced_accuracy - secondary_a.balanced_accuracy)
    return (
        f'pooled Δ={row.delta_b_minus_a_pp:+.4f} pp, paired 95% CI '
        f'[{row.paired_bootstrap_95ci_low_pp:+.4f}, {row.paired_bootstrap_95ci_high_pp:+.4f}] pp; '
        f'fold directions +/−/= {positive}/{negative}/{ties}; changed predictions {int(row.prediction_changed_n)}, '
        f'correctness discordants {int(row.correctness_discordant_n)}; '
        f'Macro-F1 Δ={secondary_delta:+.4f} pp, balanced-accuracy Δ={balanced_delta:+.4f} pp.'
    )


def main() -> None:
    config = load_config()
    replicates = int(config['reporting']['bootstrap_replicates'])
    seed = int(config['reporting']['bootstrap_seed'])
    frames = {variant: load_variant(variant) for variant in VARIANTS}
    reference = frames['#0']
    for variant, frame in frames.items():
        if not np.array_equal(frame.dataset_index, reference.dataset_index) or not np.array_equal(frame.label, reference.label):
            raise ValueError(f'Cross-variant OOF mismatch: {variant}')
        variant_root = HERE / variant
        np.save(variant_root / 'oof_sample_ids.npy', frame.dataset_index.to_numpy(np.int64))
        np.save(variant_root / 'oof_labels.npy', frame.label.to_numpy(np.int64))
        np.save(variant_root / 'oof_predictions.npy', frame.prediction.to_numpy(np.int64))
        frame.to_csv(variant_root / 'oof_sample_outputs.csv', index=False)

    metric_rows = []
    for variant, frame in frames.items():
        pooled = {'variant': variant, 'display': DISPLAY[variant], 'scope': 'POOLED_OOF', 'n': len(frame)}
        pooled.update(metrics(frame.label.to_numpy(), frame.prediction.to_numpy()))
        metric_rows.append(pooled)
        for fold in range(5):
            part = frame[frame.fold == fold]
            row = {'variant': variant, 'display': DISPLAY[variant], 'scope': f'fold_{fold}', 'n': len(part)}
            row.update(metrics(part.label.to_numpy(), part.prediction.to_numpy()))
            metric_rows.append(row)
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(HERE / 'oof_metrics.csv', index=False)

    comparison_rows = []
    for a_name, b_name in [('#0', '#1'), ('#1', '#2')]:
        comparison = f'{a_name}_vs_{b_name}'
        for fold in range(5):
            a = frames[a_name][frames[a_name].fold == fold].reset_index(drop=True)
            b = frames[b_name][frames[b_name].fold == fold].reset_index(drop=True)
            comparison_rows.append(paired(a, b, comparison, fold, replicates, seed))
        comparison_rows.append(paired(frames[a_name], frames[b_name], comparison, 'POOLED_OOF', replicates, seed))
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(HERE / 'paired_comparisons.csv', index=False)

    pooled_metrics = metric_frame[metric_frame.scope == 'POOLED_OOF'].set_index('variant')
    q_rows = {}
    for comparison, a_name, b_name in [('#0_vs_#1', '#0', '#1'), ('#1_vs_#2', '#1', '#2')]:
        pooled_row = comparisons[(comparisons.comparison == comparison) & (comparisons.scope == 'POOLED_OOF')].iloc[0]
        fold_rows = comparisons[(comparisons.comparison == comparison) & (comparisons.scope != 'POOLED_OOF')]
        q_rows[comparison] = judgement(pooled_row, fold_rows, pooled_metrics.loc[a_name], pooled_metrics.loc[b_name])

    table_lines = ['| variant | OOF accuracy | Macro-F1 | Balanced accuracy |', '|---|---:|---:|---:|']
    for variant in VARIANTS:
        row = pooled_metrics.loc[variant]
        table_lines.append(f'| {DISPLAY[variant]} | {100*row.accuracy:.4f}% | {100*row.macro_f1:.4f}% | {100*row.balanced_accuracy:.4f}% |')
    lines = [
        '# Phase 2B-S1 controlled screen', '',
        '## Completion and protocol', '',
        'All and only #0/#1/#2 completed five folds with `split_random_state=42` and `training_seed=42` independently recorded for every fold. The primary series is deterministic original-view pooled held-out OOF. This is a single-training-seed screen, not independent-seed confirmation.', '',
        *table_lines, '',
        '## Q1 — #0 vs #1: deep/narrow head architecture', '',
        q_rows['#0_vs_#1'], '',
        'Interpretation must combine the effect, interval, fold direction, changed samples, Macro-F1 and balanced accuracy above; no fixed 0.1 pp or p<0.05 gate was imposed.', '',
        '## Q2 — #1 vs #2: shortcut scaling only', '',
        q_rows['#1_vs_#2'], '',
        '#1 and #2 have the same tensor graph, module/parameter counts, mean aggregation, optimizer path and data protocol; the declared forward difference is only fixed shortcut λ=1.0 versus 0.1.', '',
        '## Mandatory stop gate', '',
        '**STOPPED_AFTER_#0_#1_#2.** No #3/#4/#5 and no new DFAG training was started. Human review is required before any continuation to MHSA, scaling+MHSA, or terminal identity experiments.', '',
        'Detailed per-fold metrics are in `oof_metrics.csv`; paired effect, bootstrap CI, exact McNemar, changed/discordant counts and fold directions are in `paired_comparisons.csv`.',
    ]
    (HERE / '00_phase2b_screen_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary = {
        'status': 'STOPPED_AFTER_#0_#1_#2',
        'training_seed': 42,
        'split_random_state': 42,
        'rows': 18353,
        'q1_evidence': q_rows['#0_vs_#1'],
        'q2_evidence': q_rows['#1_vs_#2'],
        'human_decision_required': True,
        'forbidden_training_started': False,
    }
    (HERE / 'screen_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
