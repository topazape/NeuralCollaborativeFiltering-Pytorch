from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from neural_collaborative_filtering.utils import AverageMeter, get_logger
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: Any,
        optimizer: Any,
        metric_fn: Any,
        device: str,
        save_dir: str,
    ) -> None:
        self.epochs = epochs
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.device = device
        self.save_dir = save_dir

        self.logger = get_logger(str(Path(self.save_dir).joinpath("log.txt")))
        self.best_loss = float("inf")

    def fit(self, model: nn.Module) -> None:
        for epoch in range(self.epochs):
            model.train()
            losses = AverageMeter("train_loss")
            metrics = AverageMeter("train_metric")

            with tqdm(self.train_loader, dynamic_ncols=True) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.epochs}")

                for tr_data in pbar:
                    user_idxs = tr_data[0].to(self.device)
                    item_idxs = tr_data[1].to(self.device)

                    self.optimizer.zero_grad()
                    inners = model(user_idxs, item_idxs)
                    scores = tr_data[2].float().to(self.device)

                    loss = self.criterion(inners, scores)
                    loss.backward()
                    self.optimizer.step()

                    losses.update(value=loss.item())
                    metrics.update(value=self.metric_fn(inners, scores))
                    pbar.set_postfix({"loss": losses.value})

            self.logger.info(
                f"(train) epoch: {epoch} loss: {losses.avg} rmse: {metrics.avg}"
            )
            self.evaluate(model, epoch=epoch)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, epoch: Optional[int] = None):
        model.eval()
        losses = AverageMeter("valid_loss")
        metrics = AverageMeter("valid_rmse")

        for va_data in tqdm(self.valid_loader):
            user_idxs = va_data[0].to(self.device)
            item_idxs = va_data[1].to(self.device)

            inners = model(user_idxs, item_idxs)
            scores = va_data[2].float().to(self.device)

            loss = self.criterion(inners, scores)
            losses.update(value=loss.item())
            metrics.update(value=self.metric_fn(inners, scores))

        self.logger.info(
            f"(vaid) epoch: {epoch} loss: {losses.avg} rmse: {metrics.avg}"
        )

        if epoch is not None:
            if losses.avg <= self.best_loss:
                self.best_acc = losses.avg
                torch.save(model.state_dict(), Path(self.save_dir).joinpath("best.pth"))
