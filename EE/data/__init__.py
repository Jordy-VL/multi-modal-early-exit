import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collate_fn(batch):
    batch = {
        k: torch.stack([dic[k] for dic in batch]) for k in batch[0]
    }  # List of dictionaries to dict of lists.
    return batch


# just get a custom trainer that does not mess up the columns for forward
class CustomTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=True,
        )

    def get_eval_dataloader(self, eval_dataset: Dataset) -> DataLoader:
        return DataLoader(
            self.eval_dataset if eval_dataset is None else eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return DataLoader(
            self.test_dataset if test_dataset is None else test_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )
