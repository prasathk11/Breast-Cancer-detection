import logging

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from . import c

log = logging.getLogger(__name__)


class Module(pl.LightningModule):
    def __init__(self):
        super().__init__()

        n_classes = 2

        # model
        if c.model == "efficientnet-b5":
            from efficientnet_pytorch import EfficientNet

            model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=2)
        elif c.model == "resnet18":
            model = torchvision.models.resnet18(pretrained=False, progress=True)
            model.fc = nn.Linear(512, n_classes)

        self.loss = torch.nn.CrossEntropyLoss()
        self.model = model
        self.softmax = torch.nn.Softmax(dim=-1)

        # metrics
        self.precision_ = pl.metrics.Precision()
        self.recall = pl.metrics.Recall()
        self.f1 = pl.metrics.F1(n_classes)
        self.precisionx = pl.metrics.Precision(n_classes, average=None)
        self.recallx = pl.metrics.Recall(n_classes, average=None)

    def summarize(self, *args, **kwargs):
        """ suppress message on every fit """
        pass

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """train one batch """
        x, y = batch
        # class models use tensor not list
        x = torch.stack(x)
        y = torch.stack([y["labels"][0] for y in y])

        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """ log validation losses """
        x, y = batch
        # class models use tensors not lists; y=image class
        x = torch.stack(x)
        y = torch.stack([y["labels"][0] for y in y])

        # loss
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss)

        # predict
        ypred = torch.argmax(logits, dim=1)

        # log metrics and update/compute/reset automatically
        self.log("val_f1", self.f1(ypred, y))
        self.log("val_precision", self.precision_(ypred, y))
        self.log("val_recall", self.recall(ypred, y))

        # update manually for multi-class
        self.precisionx.update(ypred, y)
        self.recallx.update(ypred, y)

        return loss

    def on_validation_epoch_end(self) -> None:
        # compute and reset manually for multi-class
        classes = ["norm/benign", "malignant"]
        for i, x in enumerate(self.precisionx.compute()):
            self.log(f"val_precision_{classes[i]}", x)
        for i, x in enumerate(self.recallx.compute()):
            self.log(f"val_recall_{classes[i]}", x)
        self.precisionx.reset()
        self.recallx.reset()

    def configure_optimizers(self):
        params = [k for k in self.parameters() if k.requires_grad]

        if c.optimizer == "onecycle_adam":
            steps_per_epoch = len(self.train_dataloader()) // c.cycles
            optimizer = torch.optim.AdamW(params, lr=c.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=c.lr,
                pct_start=c.pct_start,
                steps_per_epoch=steps_per_epoch,
                epochs=c.epochs,
            )
            scheduler = dict(scheduler=scheduler, interval="step")
            return [optimizer], [scheduler]

        if c.optimizer == "onecycle_sgd":
            optimizer = torch.optim.SGD(
                params, lr=c.lr, momentum=c.momentum, weight_decay=c.weight_decay
            )
            steps_per_epoch = len(self.train_dataloader()) // c.cycles
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=c.lr,
                steps_per_epoch=steps_per_epoch,
                epochs=c.epochs,
            )
            scheduler = dict(scheduler=scheduler, interval="step")
            return [optimizer], [scheduler]

        if c.optimizer == "cosine":
            optimizer = torch.optim.AdamW(params, lr=c.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, len(self.train_dataloader())
            )
            scheduler = dict(scheduler=scheduler, interval="step")
            return [optimizer], [scheduler]

        if c.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=c.lr, momentum=c.momentum, weight_decay=c.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=3, gamma=0.1
            )
            return [optimizer], [scheduler]

        if c.optimizer == "warmup_sgd":
            optimizer = torch.optim.SGD(
                params, lr=c.lr, momentum=c.momentum, weight_decay=c.weight_decay
            )
            steps_per_epoch = len(self.train_dataloader()) // c.cycles
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=c.lr,
                steps_per_epoch=steps_per_epoch,
                epochs=c.epochs,
                pct_start=c.pct_start,
                anneal_strategy="linear",
            )
            scheduler = dict(scheduler=scheduler, interval="step")
            return [optimizer], [scheduler]
