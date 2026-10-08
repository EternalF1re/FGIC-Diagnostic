# Historical results publication plan

Goal: publish verifiable historical predictions and statistics without retraining or editing source artifacts.

Architecture: read-only local exporter writes portable CSV OOF sets and sanitized provenance copies into paper_artifacts. An independent CPU verifier checks file hashes, exact sample/fold alignment, metrics, seed summaries, and historical paired statistics. Public artifacts record the original hash separately from any sanitized public hash.

Spec: user-supplied publication task of 2026-10-08; internal original is retained outside the public repository.

Constraints: no images, checkpoints, private paths, credentials, simulated results or silent replacement of historical numbers. Latest main manuscript/SI is not present; section-level mapping is provisional.

- [x] Discover and inventory formal sources, distinguishing diagnostic training and aborted/smoke runs.
- [x] Write adversarial verifier tests and observe failure before implementation.
- [x] Export identified complete OOF and portable copies of configs, statistical tables, manifests, logs, and historical statistical source code. Save absolute source paths in an internal-only audit.
- [x] Implement independent metrics/paired/hash verifier, preserving original RNG/category conventions; compare archived numbers and retain any differences.
- [x] Document provenance, table/section mapping, dataset access, omitted materials and all gaps. Update README and CI.
- [x] Run full tests, lint, full artifact verification and privacy scan. Save raw command output.
- [ ] Commit and push, create a versioned tag; produce an audit ZIP and publish a GitHub Release if the configured credentials permit it. No DOI claim without actual archival.

Review focus: duplicate/missing samples, fold mismatch across pairs, invalid labels, manifest path traversal, changed bytes, metric drift, and unverified source/config identities must fail visibly.
