"""Strict full-per-rank-batch wrapper for the formal CAL runner."""
from __future__ import annotations

import formal_cal_job as base


_original_loaders = base.runtime.loaders


class FullBatchLoader:
    def __init__(self, loader, batch_size: int) -> None:
        self.loader = loader
        self.batch_size = batch_size
        self.sampler = loader.sampler
        self.pin_memory = False

    def __len__(self) -> int:
        return len(self.loader.dataset) // self.batch_size

    def __iter__(self):
        yielded = 0
        for batch in self.loader:
            if int(batch[0].shape[0]) != self.batch_size:
                continue
            yield batch
            yielded += 1
            if yielded == len(self):
                return


def strict_loaders(*args, **kwargs):
    train_loader, heldout_loader, development, train_indices, heldout_indices = _original_loaders(*args, **kwargs)
    return FullBatchLoader(train_loader, 8), heldout_loader, development, train_indices, heldout_indices


def main() -> None:
    base.runtime.loaders = strict_loaders
    base.main()


if __name__ == "__main__":
    main()
