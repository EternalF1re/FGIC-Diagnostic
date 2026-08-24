# Public Release Audit

Audit date: 2026-08-24

## Status

READY FOR GIT INITIALIZATION AND GITHUB UPLOAD

## Closed release blockers

- maintained package imports are independent of the historical experiment tree;
- dataset paths are supplied only through a local manifest config;
- ResNet-50 and Inception-ResNet-v2 feature dimensions are parameterized;
- stable CLI entry points cover training, CAL DDP, validation, OOF aggregation and paired statistics;
- formal group launchers are provided under `reproduce`;
- MIT license, third-party notices, dataset boundary and contribution guidance are present;
- checkpoints, arrays, datasets, outputs, queues and logs are excluded by `.gitignore`;
- historical scripts remain only under `reference/frozen_sources` for traceability.

## Publication boundary

The repository contains source/config/documentation only. It does not include datasets, formal checkpoints or unpublished raw result arrays. The `reference/frozen_sources` directory is an audit snapshot and is not a supported execution entry point.

