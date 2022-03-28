#The code is taken from:
# https://discuss.pytorch.org/t/combine-multiple-datalaoders-for-multi-task-learning/105273
# It is modified as it's suggested in the comments

import random
import torch

class MultitaskDataLoader(torch.utils.data.DataLoader):
    def __init__(self, task_names, datasets):
        self.task_names = task_names
        self.lengths = {name : len(d) for name, d in zip(task_names, datasets)}
        indices = [[i] * v for i, v in enumerate(self.lengths.values())]
        self.task_indices = sum(indices, [])
        self.datasets = datasets

    def _reset(self):
        random.shuffle(self.task_indices)
        self.current_index = 0

    def __iter__(self):
        self.iterators = [iter(d) for d in self.datasets]
        self._reset()
        return self

    def __len__(self):
        return sum(self.lengths.values())

    def __next__(self):
        if self.current_index < len(self.task_indices):
            task_index = self.task_indices[self.current_index]
            task_name = self.task_names[task_index]
            batch = next(self.iterators[task_index])
            new_batch = (batch, task_name)
            self.current_index += 1
            return new_batch
        else:
            raise StopIteration