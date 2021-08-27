import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    EarlyStopping,
)
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.utilities import rank_zero_only

from . import c, chdir

log = logging.getLogger(__name__)


class ModelCheckpoint(ModelCheckpoint):
    """extension to reduce filesize to zero before deleting. otherwise deleted checkpoints use space in trash """

    @rank_zero_only
    def _del_model(self, filepath: str):
        if self._fs.exists(filepath):
            open(filepath, "w").close()
            os.remove(filepath)
            log.debug(f"Removed checkpoint: {filepath}")


def get_tester(**kwargs):
    """return trainer without logging or checkpoints. used for debugging, tuning and prediction.
    :param kwargs: passed to trainer.
    """
    seed_everything(c.seed)

    trainerargs = dict(
        gpus=torch.cuda.device_count(),
        benchmark=True,
        progress_bar_refresh_rate=0,
        callbacks=[ProgressBar()],
    )
    trainerargs.update(kwargs)
    return pl.Trainer(**trainerargs)


@chdir(Path(__file__).parent)
def get_trainer(**kwargs):
    """return trainer
    log config, losses, gitinfo, source files, learning rate

    :param kwargs: passed to trainer.
    """
    seed_everything(c.seed)

    # setup logger
    logger = []
    neptune_key = None
    try:
        with open(f"{c.home}/.neptune/creds.txt") as f:
            neptune_key = f.read()
    except FileNotFoundError:
        log.warning(
            "~/.neptune/creds.txt not found. continuing without neptune logging"
        )
    if neptune_key:
        neptune = NeptuneLogger(
            api_key=neptune_key,
            project_name=c.project_name,
            params=c,
            upload_source_files=["**/*.py", "**/*.yaml", "**/*.toml"],
        )
        logger.append(neptune)

    # checkpoint evey epoch to enable stop/resume
    checkpoint = ModelCheckpoint(
        dirpath=c.outputs,
        filename=f"{c.project_name.split('/')[-1]}_{neptune.version}_{{epoch}}-{{val_loss:.2f}}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
    )
    earlystopping = EarlyStopping("val_loss", patience=5)

    # defaults
    trainerargs = dict(
        default_root_dir=c.outputs,
        gpus=torch.cuda.device_count(),
        benchmark=True,
        progress_bar_refresh_rate=0,
        logger=[neptune],
        callbacks=[
            # earlystopping,
            checkpoint,
            ProgressBar(),
            LearningRateMonitor(logging_interval="step"),
        ],
        max_epochs=c.epochs,
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
        num_sanity_val_steps=2,
        gradient_clip_val=0.5,
        precision=16 if torch.cuda.device_count() else 32,
    )
    trainerargs.update(kwargs)

    return pl.Trainer(**trainerargs)


def train():
    """train pytorch lightning model """
    from .datamodule import DataModule
    from .module import Module

    model = Module()
    data = DataModule()
    trainer = get_trainer()
    trainer.fit(model, datamodule=data)
