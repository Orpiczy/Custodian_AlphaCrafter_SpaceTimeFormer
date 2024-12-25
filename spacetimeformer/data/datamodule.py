import warnings

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int,
        collate_fn=None,
        overfit: bool = False,
        pin_memory: bool = True,

    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size
        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        if overfit:
            warnings.warn("Overriding val and test dataloaders to use train set!")
        self.overfit = overfit
        self.pin_memory = pin_memory

        # # CUSTOM CODE
        # self.eval_dataset = "test"
        # self._custom_val_dataloader = None

    # def set_eval_dataset(self, eval_dataset: str) -> "DataModule":
    #     match eval_dataset:
    #         case "test":
    #             self.eval_dataset = "test"
    #         case "val":
    #             self.eval_dataset = "val"
    #         case "train":
    #             self.eval_dataset = "train"
    #         case _:
    #             raise ValueError(f"Invalid eval_dataset: {eval_dataset}")
    #     return self

    def train_dataloader(self):
        return self._make_dloader("train")

    def val_dataloader(self):
        return self._make_dloader("val")

    def test_dataloader(self):
        # return self._make_dloader(self.eval_dataset)
        return self._make_dloader("test")

    def _make_dloader(self, split):
        # if self.overfit:
        #     split = "train"
        #     shuffle = True

        return DataLoader(
            self.datasetCls(**self.dataset_kwargs, split=split),
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
        )

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )
        parser.add_argument(
            "--overfit",
            action="store_true",
        )
