class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


import os
import logging

log = logging.getLogger(__name__)

CONFIG = dotdict()
c = CONFIG

# datamodule
c.batch_size = 8
c.augment = True
c.augment_notes = ""
c.modeltype = "class"
c.model = "resnet18"
c.exclude_normal_test = False  # from testing

# module
c.optimizer = "onecycle"
c.weight_decay = 0.0005
c.momentum = 0.9
c.cycles = 1
c.pct_start = 0.5
c.notes = ""

# trainer
c.lr = 1e-4
c.epochs = 30
c.seed = 0
c.project_name = "simonm3/envisionclass"

# system dependent
if os.path.exists("c:/"):
    # windows
    c.workers = 0
    c.home = "C:/Users/simon"
    c.root = "C:/Users/simon/OneDrive/Documents/py/live/envision"
else:
    # colab
    c.workers = os.cpu_count()
    c.home = "/content/drive/MyDrive/colab"
    c.root = "/content/drive/MyDrive/colab/envision"

# input data
c.data = f"{c.root}/_data"
# model checkpoints
c.outputs = f"{c.root}/_outputs"
