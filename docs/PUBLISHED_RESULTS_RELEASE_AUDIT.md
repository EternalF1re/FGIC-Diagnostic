# Published-results release audit

Release tag: `v1.1.0-paper-artifacts`. Publication date: 2026-10-08.
Maintained package version: 1.0.0; this tag adds historical results and verification tooling, not a changed training algorithm. The definitive commit is the commit referenced by the release tag. The release asset includes a receipt recording its full commit SHA.

## Verified scope

- 134 complete five-fold OOF sets, 2,118,934 prediction rows, covering Leaves, CUB, Cars and Flowers in distinct frozen experiment contexts.
- 2,376 manifest-enumerated files plus `MANIFEST.csv`, about 97.8 MB uncompressed.
- 438 archived pooled metric assertions, 96 paired comparisons with 100,000 bootstrap draws each, 84 independent-seed mean/sample-SD assertions, and 130 archived training/held-out split checks.
- 132 OOF configuration source references and 24 maintained portable config hashes verified. Two alpha=0.5 comparator configuration references remain missing and are explicitly documented.
- 640 historical recorded config/code hashes matched original local source bytes before publication. Redacted public copies have separately recorded hashes; matching a source hash is local provenance evidence, not proof available solely from a redacted copy.
- Full recomputation: PASS, zero differences outside 1e-12 tolerance. Historical numbers and independently computed numbers are retained separately in `publication_validation/full_verification.json`.
- Unit tests: 25 passed, 0 failed in the final run. Ruff: PASS. Artifact privacy scan: 2,377 files, 928 structured JSON files, zero findings. Pattern-based scanning cannot prove absence of every possible secret.

No training, GPU profiling, experiment-output replacement or checkpoint reselection was performed. Original source hashes were checked again after export. Public copies intentionally redact private paths/identities and transcode legacy logs.

## Raw validation records and commands

See `publication_validation/pytest.txt`, `ruff.txt`, `verification_stdout.txt`, `full_verification.json`, `privacy_scan.json`, and `environment.json`.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests scripts/verify_published_results.py scripts/export_historical_results.py
python scripts/verify_published_results.py --artifact-root paper_artifacts --output verification.json
```

Local tests used `PYTHONPATH=src` rather than reinstalling the package; CI installs the editable package. Development checks included expected failures before implementation of the new provenance checks, and one collection failure without the source-layout import path. These were not final passing-test results. No original experimental code was changed to address them.

The exporter needs the owner's real historical workspace and cannot reconstruct omitted originals from this public repository. Its `--internal-audit` must be outside the public repository; private identity strings can be supplied using repeated `--redact-identity` arguments. Review sanitization before publishing a new export.

## Publication limitations

The current submitted main manuscript/SI was not available for exact table-number/text-rounding reconciliation; the mapping is experiment/section-level. Diagnostic summaries are real historical outputs, but large feature tensors, logits, checkpoints and dataset images are omitted. See `REPRODUCIBILITY_GAPS.md` and `PAPER_TABLE_MAPPING.md`. No DOI is asserted.

The GitHub release ZIP contains the complete committed public source, artifacts, environment and test records, plus an audit receipt, changed-file list and full Git diff against the base commit. It excludes `.git`, private source-path inventories, original images and weights. Git attributes preserve artifact bytes across platforms.
