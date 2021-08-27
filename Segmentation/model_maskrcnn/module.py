import logging

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from . import c

log = logging.getLogger(__name__)


class Module(pl.LightningModule):
    def __init__(self):
        super().__init__()

        num_classes = 3
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=False)

        # roi head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # mask head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        self.model = model

    def summarize(self, *args, **kwargs):
        """ suppress message on every fit """
        pass

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """train one batch """
        x, y = batch
        loss = self(x, y)
        loss["loss"] = sum(list(loss.values()))
        for k, v in loss.items():
            self.log(k, v)

        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """ log validation losses """
        x, y = batch

        # use training mode to get validation losses rather than preds
        self.train()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        with torch.no_grad():
            loss = self(x, y)

        loss["loss"] = sum(list(loss.values()))
        for k, v in loss.items():
            self.log(f"val_{k}", v, prog_bar=True)

        return loss["loss"]

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
