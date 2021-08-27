import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import *

from . import c, chdir

log = logging.getLogger(__name__)


def collate_fn(batch):
    return tuple(zip(*batch))


class DataModule(pl.LightningDataModule):
    @chdir(f"{c.data}/indexes")
    def prepare_data(self):
        """create datasets. lightning callback.

        same datamodule and datasets for class/box/mask models
        """

        # model will store test predictions here
        self.ypred = []

        # get index of data
        images = pd.read_pickle("images.pkl")
        items = pd.read_pickle("items.pkl")

        normal = images.class_ == "normal"
        train = images.split_ == "train"
        valid = images.split_ == "valid"
        test = images.split_ == "test"

        if c.modeltype == "mask":
            images = images[(images.usage == "mask")]
            items = items[items.path.isin(images.path)]
        elif c.modeltype == "box":
            images = images[images.usage.isin(["box", "mask"])]
            items = items[items.path.isin(images.path)]
        elif c.modeltype == "class":
            pass
        else:
            raise Exception("invalid modeltype")

        # may want to exclude if expected inputs always include a tumor
        if c.exclude_normal_test:
            self.eval = Imageds(images[valid & ~normal], items)
            self.tst = Imageds(images[test & ~normal], items)
        else:
            self.eval = Imageds(images[valid], items)
            self.tst = Imageds(images[test], items)

        # exclude normal from training for mask/box
        if c.modeltype == "class":
            self.trn = Imageds(images[train], items, c.augment, train=True)
            self.val = Imageds(images[valid], items)
        else:
            self.trn = Imageds(images[train & ~normal], items, c.augment, train=True)
            self.val = Imageds(images[valid & ~normal], items)

    def train_dataloader(self):
        return DataLoader(
            self.trn,
            shuffle=True,
            batch_size=c.batch_size,
            num_workers=c.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=1,
            num_workers=c.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def eval_dataloader(self):
        return DataLoader(
            self.eval,
            batch_size=1,
            num_workers=c.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tst,
            batch_size=1,
            num_workers=c.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


class Imageds(Dataset):
    def __init__(self, images, items, augment=False, train=False):
        """
        :param images: dataframe
        :param items dataframe
        :param augment: True to apply augmentation
        :param train: True for training set so bad images can be replaced
        """
        self.images = images
        self.items = items
        self.train = train

        if augment:
            # apply to image and mask
            self.augment = Compose(
                [
                    # ultrasound is always skin top => cannot use rotation, shear, perspective, verticalflip
                    RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                    RandomHorizontalFlip(),
                ]
            )
            # apply to image but not mask
            self.augment_img = Compose(
                [
                    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    # GaussianBlur(5, (10, 20)),
                ]
            )
        else:
            self.augment = False

        # PIL2tensor. note model does normalisation.
        if c.modeltype == "class":
            self.transform = Compose(
                [
                    ToTensor(),
                    # todo is this right size. same as maskrcnn
                    Resize((800, 800)),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            # maskrcnn normalises and resizes
            self.transform = Compose([ToTensor()])

    @chdir(c.data)
    def __getitem__(self, index):
        """return single item
        includes model requirements plus raw_ and aug_ for view/debug"""

        def get_boxes(masks):
            """ return boxes for the masks """
            boxes = []
            for m in masks:
                coords = np.argwhere(m > 0)
                x0, y0 = coords.min(axis=0)
                x1, y1 = coords.max(axis=0)
                # swap array coordinates to get image coordinates
                boxes.append(np.array((y0, x0, y1, x1)))
            return boxes

        # get data
        row = self.images.iloc[index]
        items = self.items.loc[self.items.path == row.path]
        img = Image.open(row.path).convert("RGB")
        masks = [
            Image.open(m).convert("L") for m in items[items.maskpath.notnull()].maskpath
        ]

        # save raw_
        raw_img = np.array(img)
        raw_masks = [np.array(m) for m in masks if np.array(m).max() > 0]
        raw_boxes = get_boxes(raw_masks)

        # augment
        if self.augment:
            # apply same random seed to augment image and masks
            seed = np.random.randint(1e6)
            torch.manual_seed(seed)
            img = self.augment(img)
            img = self.augment_img(img)
            for i, mask in enumerate(masks):
                torch.manual_seed(seed)
                masks[i] = self.augment(mask)
        masks = [np.array(m) for m in masks if np.array(m).max() > 0]
        boxes = get_boxes(masks)

        # data additional to image/boxes/masks
        classes = ["normal", "benign", "malignant"]
        labels = [classes.index(class_) for class_ in items.class_]
        if c.modeltype == "class":
            # use image class_ and reduce to 0=benign, 1=malignant
            labels = classes.index(row.class_)
            if labels > 0:
                labels = labels - 1
            labels = [labels]
        iscrowd = [False] * len(items)
        area = [(x1 - x0) * (y1 - y0) for (x0, y0, x1, y1) in boxes]
        image_id = row.name
        basename = os.path.basename(row.path)

        # save aug_
        aug_img = np.array(img).copy()
        aug_boxes = boxes.copy()
        aug_masks = masks.copy()
        aug_labels = labels.copy()

        # reject for training if target has no boxes or a box with zero area. this is common after augmentation.
        # do not reject valid/test "normal" images which do not have boxes/masks
        if (
            (c.modeltype in ["box", "mask"])
            and self.train
            and (len(boxes) == 0 or any([a <= 0 for a in area]))
        ):
            log.warning(f"refetching {row.name} as no boxes or box with zero area")
            return self[np.random.randint(len(self) - 1)]

        # convert to tensor
        img = self.transform(img)
        masks = [m > 0 for m in masks]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.as_tensor(image_id, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        tgt = dict(  # raw inputs
            raw_img=raw_img,
            raw_masks=raw_masks,
            raw_boxes=raw_boxes,
            # after augment
            aug_img=aug_img,
            aug_boxes=aug_boxes,
            aug_masks=aug_masks,
            aug_labels=aug_labels,
            # after tensor
            masks=masks,
            boxes=boxes,
            labels=labels,
            area=area,
            iscrowd=iscrowd,
            image_id=image_id,
            basename=basename,
        )
        return img, tgt

    def __len__(self):
        return len(self.images)

    def show(self, i):
        """ display image and augmented image """
        # not required for training
        from envision.eval import visualise

        visualise.main()

        _, tgt = self[i]

        # raw image
        out = []
        img = Image.fromarray(tgt["raw_img"])
        for mask in tgt["raw_masks"]:
            img = img.add_mask(mask)
        out.append(img)

        # augmented image
        if (tgt["aug_img"] == tgt["raw_img"]).all():
            log.warning("showing original image only as there is no augmented image")
            return
        img = Image.fromarray(tgt["aug_img"])
        for mask in tgt["aug_masks"]:
            img = img.add_mask(mask)
        out.append(img)

        return out

    def showid(self, image_id):
        """ return original and augmented image from index """
        i = self.images.index.get_loc(image_id)
        return self.show(i)
