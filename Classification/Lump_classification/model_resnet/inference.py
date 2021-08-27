import logging
from copy import deepcopy
from typing import Dict

import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm.auto import tqdm

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
        res = self.predict(image)

        # reformat for pipeline
        res["label"] = "malignant" if res["class_"] == 1 else "clear"
        return res

    @torch.no_grad()
    def predict(self, image):
        """predict ypred for single image

        :param image: numpy array
        :return: ypred=[dict]. includes image class_ and prob."""
        # preprocess
        transform = Compose(
            [
                ToTensor(),
                Resize((800, 800)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        x = [transform(image)]
        x = torch.stack(x)
        x = x.to(self.model.device)

        # model
        logits = self.model(x)[0]

        # postprocess
        ypred = dict(
            class_=torch.argmax(logits).item(),
            prob=self.model.softmax(logits)[1].item(),
        )
        ypred = self.postprocess([ypred])[0]
        return ypred

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader):
        """evaluation of trained model. excludes post-processing so this can be tuned separately.
        train logs losses and metrics
        valid logs losses and metrics

        :param inputs: iterable([x],[y]) for evaluation
        :return: ytrue=[dict], ypred=[class_, prob, masks, boxes, labels, scores]
        """
        ytrue = []
        ypred = []
        self.model.eval()
        for x, y in tqdm(dataloader):
            x = torch.stack(x)
            x = x.to(self.model.device)
            # ypred. class is argmax. prob(malignant) is column 1
            logits = self.model(x)
            class_ = torch.argmax(logits, dim=1)
            prob = self.model.softmax(logits)[:, 1]
            ypred.extend(
                [dict(class_=c.item(), prob=p.item()) for c, p in zip(class_, prob)]
            )

            # ytrue. includes image for display
            for y1 in y:
                y1["class_"] = y1["labels"].max().item()
                ytrue.append(y1)

            # postprocessing separate for tuning

        return ytrue, ypred

    def postprocess(self, ypred, roc_cutoff=0.5):
        ypred = deepcopy(ypred)
        for i, y in enumerate(ypred):
            y = {
                k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in y.items()
            }

            # improves malignant recall but increases false positives.
            if y["prob"] > roc_cutoff:
                y["class_"] = 1

            # convert to numpy
            y = {
                k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                for k, v in y.items()
            }
            ypred[i] = y

        return ypred
