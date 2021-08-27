import logging
from copy import deepcopy
from typing import Dict

import torch
from torchvision.ops import nms
from torchvision.transforms import Compose, ToTensor
from tqdm.auto import tqdm
import numpy as np

from .module import Module

log = logging.getLogger()


class InferenceModel:
    def __init__(self, weights):
        model = Module()
        model.load_state_dict(torch.load(weights))
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.to(device)
        model.eval()
        self.model = model

    def __call__(self, image) -> Dict:
        """ ONLY FOR PIPELINE. FOR EVALUATION JUST CALL PREDICT """
        res = self.predict(image)

        # reformat for pipeline
        m = res["masks"].squeeze(1).sum(axis=0)
        m = np.clip(m, 0, 1)
        res["mask"] = (m * 255).astype(np.uint8)
        res["masks"] = [(m.squeeze(0) * 255).astype(np.uint8) for m in res["masks"]]
        textlabels = {0: "normal", 1: "benign", 2: "malignant"}
        res["labels"] = [textlabels[x] for x in res["labels"]]
        res["label"] = "malignant" if res["class_"] == 1 else "clear"
        res["boxes"] = [x.tolist() for x in res["boxes"]]
        res["scores"] = [x for x in res["scores"]]

        return res

    @torch.no_grad()
    def predict(self, image) -> Dict:
        """predict ypred for list of images

        :param image: numpy array
        :return: ypred=[dict]. includes image class_ and prob."""
        # preprocess
        transform = Compose([ToTensor()])
        x = transform(image)
        x = x.to(self.model.device)

        # model
        ypred = self.model([x])[0]

        # postprocess
        ypred = self.postprocess([ypred])[0]
        return ypred

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader):
        """get ytrue and ypred for dataloader. excludes post-processing so this can be tuned separately.
        train logs losses and metrics
        valid logs losses and metrics

        :param inputs: iterable(x,y)
        :return: ytrue=[dict], ypred=[dict]. includes image class_
        """
        ytrue = []
        ypred = []
        for x, y in tqdm(dataloader):
            x = torch.stack(x)
            x = x.to(self.model.device)

            # model
            res = self.model(x)
            ypred.extend(res)

            # ytrue
            for y1 in y:
                y1["class_"] = (
                    0 if len(y1["labels"]) == 0 else y1["labels"].max().item()
                )
                y1["class_"] = 1 if y1["class_"] == 2 else 0
                ytrue.append(y1)

            # postprocessing separate for tuning

        return ytrue, ypred

    def coco_evaluate(self, dataloader):
        """ evaluate mAP and mAR at various IOU levels and object sizes for boxes and masks """
        # import here as only place depenncy required. coco_tools must be in sys.path due to absolute imports
        import engine

        engine.evaluate(self.model.model, dataloader, device=self.model.device)

    def postprocess(self, ypred, maxiou=0.1, minscore=0.5, flip=0):
        """filter results for box/mask model
        filter using nms/minscore; move to numpy; set image class/prob
        :param ypred: list(dict) output from predict
        :param maxiou: where two items have iou above this then lowest scoring are removed
        :param minscore: items with score less than this are removed
        """
        # copy so can rerun without losing the original
        ypred = deepcopy(ypred)
        for i, y in enumerate(ypred):
            y = {
                k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in y.items()
            }

            # only accept items above cutoff score
            sel = torch.nonzero(y["scores"] > minscore)[:, 0]
            y = {k: v[sel] for k, v in y.items()}

            # boost malignant to be more important
            y["scores2"] = torch.Tensor(
                [
                    score + 1 if label == 2 else score
                    for label, score in zip(y["labels"], y["scores"])
                ]
            )
            # nms on all classes at once with priority to malignant
            sel = nms(y["boxes"], y["scores2"], maxiou)
            y = {k: v[sel] for k, v in y.items()}
            del y["scores2"]

            # convert benign with p < flip to malignant
            for i2, label in enumerate(y["labels"]):
                if label == 1 and y["scores"][i2] < flip:
                    y["labels"][i2] = 2
                    y["scores"][i2] = 1 - y["scores"][i2]

            # move to numpy after nms complete
            y = {
                k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                for k, v in y.items()
            }
            # image level consolidate
            y["class_"] = 0 if len(y["labels"]) == 0 else y["labels"].max()
            y["class_"] = 1 if y["class_"] == 2 else 0

            ypred[i] = y
        return ypred
