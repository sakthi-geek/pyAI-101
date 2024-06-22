import numpy as np
import random

from pyAI.autograd.tensor import Tensor

class DataLoader:           # WORK IN PROGRESS
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn if collate_fn is not None else self.default_collate_fn
        self.indices = list(range(len(dataset)))
        self._reset_indices()

    def _reset_indices(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_idx = 0

    def __iter__(self):
        self._reset_indices()
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = [self.dataset[idx] for idx in batch_indices]
        self.current_idx += self.batch_size
        return self.collate_fn(batch)

    def default_collate_fn(self, batch):
        if isinstance(batch[0], Tensor):
            return Tensor(np.stack([item.data for item in batch]))
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [self.default_collate_fn(samples) for samples in transposed]
        else:
            return np.array(batch)
