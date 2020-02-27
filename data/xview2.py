from torch.utils.data import Dataset
import glob
import os
import json
from shapely import wkt
from shapely.geometry import mapping
import numpy as np
import cv2
from PIL import Image
from simplecv.api.preprocess import comm
from simplecv.api.preprocess import segm
import torch
from simplecv.util import viz
from torchvision.transforms import functional as F
from skimage.io import imread
from albumentations import Compose


class Xview2(Dataset):
    Damage_label = {
        'un-classified': 255,
        'no-damage': 1,
        'minor-damage': 2,
        'major-damage': 3,
        'destroyed': 4
    }

    def __init__(self,
                 image_dir,
                 labels_dir,
                 mode='segm',
                 include=('pre', 'post'),
                 transforms=None):
        assert mode in ('segm',)

        self.mode = mode
        self.include = include
        if isinstance(transforms, Compose):
            # albumentations
            self.transforms = transforms
        else:
            self.transforms = comm.Compose(transforms)
        self.image_dir = image_dir
        self.labels_dir = labels_dir

        self.imfp_list = glob.glob(os.path.join(image_dir, '*.png'))
        self.imname_list = [os.path.basename(imfp) for imfp in self.imfp_list]

        self.gtpath_list = [os.path.join(labels_dir, imname.replace('png', 'json')) for imname in self.imname_list]
        self._filter()
        self._remove_empty()
        # self._grouping()

    @staticmethod
    def read_json(json_path):
        # make type
        post = False
        if 'post' in os.path.basename(json_path):
            post = True
        blob = []
        with open(json_path, 'r') as f:
            jobj = json.load(f)
            features = jobj['features']
            entry_list = features['xy']

            for e in entry_list:
                wkt_str = e['wkt']
                shape = wkt.loads(wkt_str)
                # a set of points
                coords = list(mapping(shape)['coordinates'][0])
                coords = np.array(coords, np.int32)

                prop = e['properties']

                ftype = prop['feature_type']
                assert ftype == 'building'
                if post:
                    sub_type = prop['subtype']
                    blob.append((coords, ftype, sub_type))
                else:
                    # pre disaster
                    blob.append((coords, ftype, 'no-damage'))
        return blob

    @staticmethod
    def make_mask(size, points_set, labels):
        mask = np.zeros(size, np.uint8)
        ret = np.zeros(size, np.uint8)
        for points, label in zip(points_set, labels):
            cv2.fillPoly(mask, [points], (255, 255, 255))
            mask = np.where(mask > 0, np.ones_like(mask), np.zeros_like(mask))

            mask *= Xview2.Damage_label[label]
            ret = np.where(ret == 0, mask, ret)

        return ret

    @staticmethod
    def viz_image_mask(img_tensor: torch.Tensor, mask_tensor: torch.Tensor, alpha=0.8):
        img = img_tensor.permute(1, 2, 0).numpy()
        mask = mask_tensor.numpy()
        mask = np.stack([mask] * 3, axis=2)
        return viz.plot_image_color_mask(img, mask * 255, alpha)

    def _grouping(self):
        u_name_list = [name.replace('post', 'coffee') for name in self.imname_list if 'post' in name]
        assert len(u_name_list) * 2 == len(self.imname_list)

    def _remove_empty(self):
        org_size = len(self.imfp_list)

        keep_inds = []
        for idx, gtpath in enumerate(self.gtpath_list):
            with open(gtpath, 'r') as f:
                jobj = json.load(f)
                features = jobj['features']
                entry_list = features['xy']
                if len(entry_list) != 0:
                    keep_inds.append(idx)

        self.imfp_list = [self.imfp_list[idx] for idx in keep_inds]
        self.gtpath_list = [self.gtpath_list[idx] for idx in keep_inds]
        self.imname_list = [self.imname_list[idx] for idx in keep_inds]

        print('[Remove Empty] data size: ', org_size, '->', len(self.imfp_list))

    def _filter(self):
        if len(self.include) == 2:
            return
        keep_inds = []
        for idx, imname in enumerate(self.imname_list):
            if any([key in imname for key in self.include]):
                keep_inds.append(idx)

        self.imfp_list = [self.imfp_list[idx] for idx in keep_inds]
        self.gtpath_list = [self.gtpath_list[idx] for idx in keep_inds]
        self.imname_list = [self.imname_list[idx] for idx in keep_inds]

    def __len__(self):
        return len(self.imfp_list)

    def __getitem__(self, idx):
        imfp = self.imfp_list[idx]
        if isinstance(imfp, tuple):
            pre_imfp, post_imfp = imfp
            pre_img = imread(pre_imfp)
            post_img = imread(post_imfp)
            prepost_img = np.concatenate([pre_img, post_img], axis=2)

            pre_gtpath, post_gtpath = self.gtpath_list[idx]

            pre_entries = Xview2.read_json(pre_gtpath)
            post_entries = Xview2.read_json(post_gtpath)

            pre_mask = Xview2.make_mask((1024, 1024), [e[0] for e in pre_entries], [e[2] for e in pre_entries])
            post_mask = Xview2.make_mask((1024, 1024), [e[0] for e in post_entries], [e[2] for e in post_entries])
            prepost_mask = np.stack([pre_mask, post_mask], axis=2)
            if self.transforms is not None:
                blob = self.transforms(**dict(image=prepost_img, mask=prepost_mask))
                img = blob['image']
                mask = blob['mask']
                return img, dict(cls=mask)
            raise NotImplementedError
        else:
            img = Image.open(imfp)
        json_path = self.gtpath_list[idx]
        entries = Xview2.read_json(json_path)
        mask = Xview2.make_mask((1024, 1024), [e[0] for e in entries], [e[2] for e in entries])
        mask = Image.fromarray(mask)

        if self.transforms.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, dict(cls=mask, image_filename=os.path.basename(imfp))

    def pairwise_mode(self):
        assert len(self.include) == 2
        u_name_list = [name.replace('post', 'coffee') for name in self.imname_list if 'post' in name]
        pre_imfp_list = [os.path.join(self.image_dir, uname.replace('coffee', 'pre')) for uname in u_name_list]
        post_imfp_list = [os.path.join(self.image_dir, uname.replace('coffee', 'post')) for uname in u_name_list]

        self.imfp_list = [(pre, post) for pre, post in zip(pre_imfp_list, post_imfp_list)]
        pre_gtfp_list = [os.path.join(self.labels_dir, uname.replace('coffee', 'pre').replace('png', 'json')) for uname
                         in u_name_list]
        post_gtfp_list = [os.path.join(self.labels_dir, uname.replace('coffee', 'post').replace('png', 'json')) for
                          uname
                          in u_name_list]

        self.gtpath_list = [(pre, post) for pre, post in zip(pre_gtfp_list, post_gtfp_list)]
        return self


class Xview2Test(Dataset):
    def __init__(self, image_dir):
        super(Xview2Test, self).__init__()
        self.image_dir = image_dir
        self.imfp_list = glob.glob(os.path.join(image_dir, '*.png'))
        self.imname_list = [os.path.basename(imfp) for imfp in self.imfp_list]
        u_name_list = [name.replace('post', 'coffee') for name in self.imname_list if 'post' in name]
        self.pre_imfp_list = [os.path.join(self.image_dir, uname.replace('coffee', 'pre')) for uname in u_name_list]
        self.post_imfp_list = [os.path.join(self.image_dir, uname.replace('coffee', 'post')) for uname in u_name_list]

    def __getitem__(self, idx):
        pre_imfp = self.pre_imfp_list[idx]
        post_imfp = self.post_imfp_list[idx]
        pre_img = Image.open(pre_imfp)
        post_img = Image.open(post_imfp)

        pre_img = 255. * F.to_tensor(pre_img)
        post_img = 255. * F.to_tensor(post_img)

        npre_img = F.normalize(pre_img, (123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
        npost_img = F.normalize(post_img, (123.675, 116.28, 103.53), (58.395, 57.12, 57.375))

        return npre_img, npost_img, os.path.basename(pre_imfp)

    def __len__(self):
        assert len(self.pre_imfp_list) == len(self.post_imfp_list)
        return len(self.pre_imfp_list)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from albumentations import Compose, OneOf, Normalize
    from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomScale, RandomCrop
    from albumentations.pytorch import ToTensorV2

    dataset = Xview2(r'D:\DATA\xView2\train\images',
                     r'D:\DATA\xView2\train\labels',
                     transforms=Compose([
                         OneOf([
                             HorizontalFlip(True),
                             VerticalFlip(True),
                             RandomRotate90(True)
                         ], p=0.5),
                         # RandomDiscreteScale([0.75, 1.25, 1.5], p=0.5),
                         RandomCrop(640, 640, True),
                         Normalize(mean=(0.485, 0.456, 0.406,
                                         0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225,
                                        0.229, 0.224, 0.225), max_pixel_value=255),
                         ToTensorV2(True),
                     ]),
                     include=('pre', 'post')).pairwise_mode()

    print(len(dataset))
    a = dataset[1]
    print()
    # img, mask = dataset[4]
    # print(np.unique(mask))
    # for e in tqdm(dataset):
    #     pass
    # viz_img = Xview2.viz_image_mask(img, mask)
    #
    # plt.imshow(viz_img)
    # plt.show()
