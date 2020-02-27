from data.xview2 import Xview2Test
from torch.utils.data import DataLoader
from tqdm import tqdm
import fire
from simplecv.api.infer_tool import build_and_load_from_file
import torch
from concurrent.futures import ProcessPoolExecutor
from module import register_model
from simplecv.core.transform_base import ParallelTestTransform
from simplecv.data.test_transforms import Rotate90k, HorizontalFlip, VerticalFlip, Transpose, Scale
import torch.nn as nn
from skimage.io import imread, imsave
from skimage import measure
import os
import numpy as np


def tta(model, image):
    trans = ParallelTestTransform(
        Rotate90k(1),
        Rotate90k(2),
        Rotate90k(3),
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        Scale(scale_factor=0.75),
        Scale(scale_factor=1.0),
        Scale(scale_factor=1.25),
        Scale(scale_factor=1.5),
    )
    images = trans.transform(image)
    with torch.no_grad():
        outs = [model(im) for im in images]

    outs = trans.inv_transform(outs)

    out = sum(outs) / len(outs)

    return out


def output_name(pre_filename: str):
    """

    Args:
        pre_filename: test_pre_*****.png

    Returns:
        test_localization_*****_prediction.png,
        test_damage_*****_prediction.png
    """
    test_local = pre_filename.replace('pre', 'localization').replace('.png', '_prediction.png')
    test_damage = pre_filename.replace('pre', 'damage').replace('.png', '_prediction.png')
    return test_local, test_damage


def localize(output_dir, config_path, checkpoint_path):
    os.makedirs(output_dir, exist_ok=True)
    dataset = Xview2Test('./xview2/test/images')
    dataloader = DataLoader(dataset, batch_size=torch.cuda.device_count(), shuffle=False, num_workers=12)
    ppe = ProcessPoolExecutor(max_workers=4)
    model, gs = build_and_load_from_file(config_path, checkpoint_path)
    model = model.cpu()
    model.to(torch.device('cuda'))
    model = nn.DataParallel(model, list(range(torch.cuda.device_count())))
    for npre_img, npost_img, pre_fname in tqdm(dataloader):
        blob = [output_name(name) for name in pre_fname]
        out_names, _ = list(zip(*blob))

        with torch.no_grad():
            npre_img = npre_img.to(torch.device('cuda'))

            prob = tta(model, npre_img)
            # prob = model(npre_img)

            binary = prob > 0.5

            binary = binary.squeeze(dim=1).cpu().numpy()

            if len(pre_fname) == 1:
                ppe.submit(imsave, os.path.join(output_dir, out_names[0]), binary[0])
            else:
                for idx, out_name in enumerate(out_names):
                    ppe.submit(imsave, os.path.join(output_dir, out_name), binary[idx])

    ppe.shutdown()
    torch.cuda.empty_cache()


def damage(output_dir, config_path, checkpoint_path):
    os.makedirs(output_dir, exist_ok=True)
    dataset = Xview2Test('./xview2/test/images')
    dataloader = DataLoader(dataset, batch_size=torch.cuda.device_count(), shuffle=False, num_workers=12)
    ppe = ProcessPoolExecutor(max_workers=4)
    model, gs = build_and_load_from_file(config_path, checkpoint_path)
    model = model.cpu()
    model.to(torch.device('cuda'))
    model = nn.DataParallel(model, list(range(torch.cuda.device_count())))
    for npre_img, npost_img, pre_fname in tqdm(dataloader):
        blob = [output_name(name) for name in pre_fname]
        _, out_names = list(zip(*blob))

        with torch.no_grad():
            npost_img = npost_img.to(torch.device('cuda'))

            prob = tta(model, npost_img)

            # [N, H, W]
            pred = prob.argmax(dim=1)
            pred = pred.cpu().numpy().astype(np.uint8)

            if len(pre_fname) == 1:
                ppe.submit(imsave, os.path.join(output_dir, out_names[0]), pred[0])
            else:
                for idx, out_name in enumerate(out_names):
                    ppe.submit(imsave, os.path.join(output_dir, out_name), pred[idx])

    ppe.shutdown()
    torch.cuda.empty_cache()


def e2e_loc_dam(loc_output_dir, dam_output_dir, config_path, checkpoint_path):
    os.makedirs(loc_output_dir, exist_ok=True)
    os.makedirs(dam_output_dir, exist_ok=True)

    dataset = Xview2Test('./xview2/test/images')
    dataloader = DataLoader(dataset, batch_size=torch.cuda.device_count(), shuffle=False, num_workers=12)
    ppe = ProcessPoolExecutor(max_workers=4)
    model, gs = build_and_load_from_file(config_path, checkpoint_path)
    model = model.cpu()
    model.to(torch.device('cuda'))
    model = nn.DataParallel(model, list(range(torch.cuda.device_count())))
    for npre_img, npost_img, pre_fname in tqdm(dataloader):
        blob = [output_name(name) for name in pre_fname]
        loc_out_names, dam_out_names = list(zip(*blob))

        with torch.no_grad():
            nprepost_img = torch.cat([npre_img, npost_img], dim=1)
            nprepost_img = nprepost_img.to(torch.device('cuda'))

            prepost_prob = tta(model, nprepost_img)
            pre_prob = prepost_prob[:, :1, :, :]
            post_prob = prepost_prob[:, 1:, :, :]
            # 1. loc
            binary = pre_prob > 0.5
            binary = binary.squeeze(dim=1).cpu().numpy()

            if len(pre_fname) == 1:
                ppe.submit(imsave, os.path.join(loc_output_dir, loc_out_names[0]), binary[0])
            else:
                for idx, out_name in enumerate(loc_out_names):
                    ppe.submit(imsave, os.path.join(loc_output_dir, out_name), binary[idx])
            # 2. dam
            # [N, H, W]
            pred = post_prob.argmax(dim=1)
            pred = pred.cpu().numpy().astype(np.uint8)

            if len(pre_fname) == 1:
                ppe.submit(imsave, os.path.join(dam_output_dir, dam_out_names[0]), pred[0])
            else:
                for idx, out_name in enumerate(dam_out_names):
                    ppe.submit(imsave, os.path.join(dam_output_dir, out_name), pred[idx])

    ppe.shutdown()
    torch.cuda.empty_cache()


def fuse_location_damage(location_dir, damage_dir, save_dir, cls_weight_list=(1., 1., 1., 1.)):
    os.makedirs(save_dir, exist_ok=True)
    damage_cls_list = [1, 2, 3, 4]

    localization_filename = 'test_localization_{}_prediction.png'
    damage_filename = 'test_damage_{}_prediction.png'

    for i in tqdm(range(933)):
        # 1. read localization mask
        local_path = os.path.join(location_dir, localization_filename.format(str(i).zfill(5)))
        local_mask = imread(local_path)
        # 2. get connected regions
        labeled_local, nums = measure.label(local_mask, connectivity=2, background=0, return_num=True)
        region_idlist = np.unique(labeled_local)
        # 3. start vote
        if len(region_idlist) > 1:
            dam_path = os.path.join(damage_dir, damage_filename.format(str(i).zfill(5)))
            dam_mask = imread(dam_path)
            new_dam = local_mask.copy()
            for region_id in region_idlist:
                # if background, ignore it
                if all(local_mask[local_mask == region_id]) == 0:
                    continue
                region_dam_count = [int(np.sum(dam_mask[labeled_local == region_id] == dam_cls_i)) * cls_weight \
                                    for dam_cls_i, cls_weight in zip(damage_cls_list, cls_weight_list)]
                # vote
                dam_index = np.argmax(region_dam_count) + 1
                new_dam = np.where(labeled_local == region_id, dam_index, new_dam)
        else:
            new_dam = local_mask.copy()

        imsave(os.path.join(save_dir, damage_filename.format(str(i).zfill(5))), new_dam)


def e2e_loc_dam_out_prob(loc_output_dir, dam_output_dir, config_path, checkpoint_path):
    os.makedirs(loc_output_dir, exist_ok=True)
    os.makedirs(dam_output_dir, exist_ok=True)

    dataset = Xview2Test('./xview2/test/images')
    dataloader = DataLoader(dataset, batch_size=torch.cuda.device_count(), shuffle=False, num_workers=12)
    ppe = ProcessPoolExecutor(max_workers=4)
    model, gs = build_and_load_from_file(config_path, checkpoint_path)
    model = model.cpu()
    model.to(torch.device('cuda'))
    model = nn.DataParallel(model, list(range(torch.cuda.device_count())))
    for npre_img, npost_img, pre_fname in tqdm(dataloader):
        blob = [output_name(name) for name in pre_fname]
        loc_out_names, dam_out_names = list(zip(*blob))

        with torch.no_grad():
            nprepost_img = torch.cat([npre_img, npost_img], dim=1)
            nprepost_img = nprepost_img.to(torch.device('cuda'))

            prepost_prob = tta(model, nprepost_img)
            pre_prob = prepost_prob[:, :1, :, :]
            post_prob = prepost_prob[:, 1:, :, :]
            # 1. loc
            binary = pre_prob.squeeze(dim=1).cpu().numpy()
            binary *= 100
            binary = binary.astype(np.uint8)

            if len(pre_fname) == 1:
                ppe.submit(np.save, os.path.join(loc_output_dir, loc_out_names[0].replace('png', 'npy')), binary[0])
            else:
                for idx, out_name in enumerate(loc_out_names):
                    ppe.submit(np.save, os.path.join(loc_output_dir, out_name.replace('png', 'npy')), binary[idx])
            # 2. dam
            # [N, H, W]
            pred = post_prob
            pred = pred.cpu().numpy()
            pred *= 100
            pred = pred.astype(np.uint8)

            if len(pre_fname) == 1:
                ppe.submit(np.save, os.path.join(dam_output_dir, dam_out_names[0].replace('png', 'npy')), pred[0])
            else:
                for idx, out_name in enumerate(dam_out_names):
                    ppe.submit(np.save, os.path.join(dam_output_dir, out_name.replace('png', 'npy')), pred[idx])

    ppe.shutdown()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    fire.Fire(dict(
        localize=localize,
        damage=damage,
        fuse_location_damage=fuse_location_damage,
        e2e_loc_dam=e2e_loc_dam,
        e2e_loc_dam_out_prob=e2e_loc_dam_out_prob
    ))
