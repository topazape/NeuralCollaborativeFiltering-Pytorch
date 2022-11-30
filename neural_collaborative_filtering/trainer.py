from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from neural_collaborative_filtering.utils import AverageMeter, get_logger
from torch.utils.data import DataLoader
from torcheval.metrics.functional import hit_rate
from tqdm import tqdm

from .metrics import dcg2


class Trainer:
    def __init__(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: Any,
        optimizer: Any,
        device: str,
        save_dir: str,
    ) -> None:
        self.epochs = epochs
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir

        self.logger = get_logger(str(Path(self.save_dir).joinpath("log.txt")))
        self.best_loss = float("inf")

    def fit(self, model: nn.Module) -> None:
        for epoch in range(self.epochs):
            model.train()
            losses = AverageMeter("train_loss")

            with tqdm(self.train_loader, dynamic_ncols=True) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.epochs}")

                for tr_data in pbar:
                    user_idxs = tr_data[0].to(self.device)
                    item_idxs = tr_data[1].to(self.device)

                    self.optimizer.zero_grad()
                    out = model(user_idxs, item_idxs)
                    target = tr_data[2].float().to(self.device)

                    loss = self.criterion(out, target)
                    loss.backward()
                    self.optimizer.step()

                    losses.update(value=loss.item())

                    pbar.set_postfix({"loss": losses.value})

            self.logger.info(f"(train) epoch: {epoch} loss: {losses.avg}")
            self.evaluate(model, epoch=epoch)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, epoch: Optional[int] = None):
        model.eval()
        losses = AverageMeter("valid_loss")
        hrs = AverageMeter("valid_HR@10")
        dcg2s = AverageMeter("valid_NDCG@10")

        for va_data in tqdm(self.valid_loader):
            user_idxs = va_data[0].to(self.device)
            item_idxs = va_data[1].to(self.device)

            input = model(user_idxs, item_idxs)
            target = va_data[2].float().to(self.device)

            loss = self.criterion(input, target)
            losses.update(value=loss.item())

            gt_item = item_idxs[0].reshape(1)
            hitrate_at_10 = hit_rate(input.reshape(1, -1), gt_item, k=10)
            hrs.update(value=hitrate_at_10.item())
            dcg2s.update(value=dcg2(input, gt_item, k=10).item())

        self.logger.info(
            f"(vaid) epoch: {epoch} loss: {losses.avg} {hrs.name}: {hrs.avg} {dcg2s.name}: {dcg2s.avg}"
        )

        if epoch is not None:
            if losses.avg <= self.best_loss:
                self.best_acc = losses.avg
                torch.save(model.state_dict(), Path(self.save_dir).joinpath("best.pth"))
