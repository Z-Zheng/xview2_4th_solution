from data.xview2 import Xview2
from torch.utils.data.dataloader import DataLoader
from simplecv.api.preprocess import comm
from simplecv.api.preprocess import segm
from simplecv import registry
from simplecv.core.config import AttrDict
from simplecv.data import distributed
from simplecv.data.cross_validation import CrossValSamplerGenerator
from torch.utils.data import SequentialSampler
from torch.utils.data import ConcatDataset
from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomScale, RandomCrop
from albumentations.pytorch import ToTensorV2
import cv2
import random


class RandomDiscreteScale(RandomScale):
    def __init__(self, scales, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5):
        super(RandomDiscreteScale, self).__init__(0, interpolation, always_apply, p)
        self.scales = scales

    def get_params(self):
        return {"scale": random.choice(self.scales)}


@registry.DATALOADER.register('Xview2DataLoader')
class Xview2DataLoader(DataLoader):
    def __init__(self, config):
        self.config = AttrDict()
        self.set_defalut()
        self.config.update(config)
        if any([isinstance(self.config.image_dir, tuple),
                isinstance(self.config.image_dir, list)]):
            dataset_list = []
            for im_dir, label_dir in zip(self.config.image_dir, self.config.label_dir):
                dataset_list.append(Xview2(im_dir,
                                           label_dir,
                                           self.config.mode,
                                           self.config.include,
                                           self.config.transforms))

            dataset = ConcatDataset(dataset_list)

        else:
            dataset = Xview2(self.config.image_dir,
                             self.config.label_dir,
                             self.config.mode,
                             self.config.include,
                             self.config.transforms)

        if self.config.CV.on and self.config.CV.cur_k != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k_fold)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.cur_k]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(Xview2DataLoader, self).__init__(dataset,
                                               self.config.batch_size,
                                               sampler=sampler,
                                               num_workers=self.config.num_workers,
                                               pin_memory=True)

    def set_defalut(self):
        self.config.update(dict(
            image_dir='',
            label_dir='',
            mode='segm',
            include=('pre', 'post'),
            CV=dict(
                on=True,
                cur_k=0,
                k_fold=5,
            ),
            transforms=[
                segm.RandomHorizontalFlip(0.5),
                segm.RandomVerticalFlip(0.5),
                segm.RandomRotate90K((0, 1, 2, 3)),
                segm.FixedPad((1024, 1024), 255),
                segm.ToTensor(True),
                comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
            ],
            batch_size=1,
            num_workers=0,
            training=True
        ))


@registry.DATALOADER.register('Xview2PairwiseDataLoader')
class Xview2PairwiseDataLoader(DataLoader):
    def __init__(self, config):
        self.config = AttrDict()
        self.set_defalut()
        self.config.update(config)
        if any([isinstance(self.config.image_dir, tuple),
                isinstance(self.config.image_dir, list)]):
            dataset_list = []
            for im_dir, label_dir in zip(self.config.image_dir, self.config.label_dir):
                dataset_list.append(Xview2(im_dir,
                                           label_dir,
                                           self.config.mode,
                                           self.config.include,
                                           self.config.transforms).pairwise_mode())

            dataset = ConcatDataset(dataset_list)

        else:
            dataset = Xview2(self.config.image_dir,
                             self.config.label_dir,
                             self.config.mode,
                             self.config.include,
                             self.config.transforms).pairwise_mode()

        if self.config.CV.on and self.config.CV.cur_k != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k_fold)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.cur_k]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(Xview2PairwiseDataLoader, self).__init__(dataset,
                                                       self.config.batch_size,
                                                       sampler=sampler,
                                                       num_workers=self.config.num_workers,
                                                       pin_memory=True)

    def set_defalut(self):
        self.config.update(dict(
            image_dir='',
            label_dir='',
            mode='segm',
            include=('pre', 'post'),
            CV=dict(
                on=True,
                cur_k=0,
                k_fold=5,
            ),
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], p=0.5),
                RandomDiscreteScale([0.75, 1.25, 1.5], p=0.5),
                RandomCrop(640, 640, True),
                Normalize(mean=(0.485, 0.456, 0.406,
                                0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225,
                               0.229, 0.224, 0.225), max_pixel_value=255),
                ToTensorV2(True),
            ]),
            batch_size=1,
            num_workers=0,
            training=True
        ))
