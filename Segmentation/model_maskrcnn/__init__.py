import contextlib
import logging
import os

from .config import c

log = logging.getLogger()

# torchvision cache of weights
os.environ["TORCH_HOME"] = f"{c.root}/.cache"


@contextlib.contextmanager
def chdir(path=None):
    """ temporary os.chdir(path) """
    saved = ""
    if path is None:
        return
    try:
        saved = os.getcwd()
    except FileNotFoundError:
        pass
        # log.warning("no current working folder")
    os.chdir(path)
    try:
        yield
    finally:
        try:
            # google drive bug cannot return to a shortcut folder
            os.chdir(saved)
        except FileNotFoundError:
            pass
            # log.warning(f"unable to chdir back to {saved}")


from .module import Module
from .datamodule import DataModule
from .training import train, get_trainer, get_tester
from .inference import InferenceModel
