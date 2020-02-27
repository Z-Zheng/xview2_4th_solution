from simplecv.data import test_transforms as ttas
from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
from simplecv.api.preprocess import albu
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

config = dict(
    model=dict(
        type='GSiameseResNet',
        params=dict(
            backbone=dict(
                resnet_type='resnext101_32x4d',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                # 16 or 32
                output_stride=32,
                with_cp=(False, False, False, False),
                norm_layer=nn.BatchNorm2d,
            ),
            neck=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
            ),
            head=dict(
                in_channels=256,
                out_channels=256,
                num_classes=5,
                upsample_scale=4.0,
                num_blocks=1,
                bottleneck_channels=128
            ),
            loss=dict(
                cls_weight=1.0,
                ignore_index=255,
                dam=dict(
                    ohem=dict(
                        ratio=0.8
                    )
                ),
                loc=dict(
                    tversky_loss=dict(alpha=0.7, beta=0.3),
                    bce_loss=dict(),
                )
            )
        ),
    ),
    data=dict(
        train=dict(
            type='Xview2PairwiseDataLoader',
            params=dict(
                image_dir=('./xview2/train/images', './xview2/tier3/images'),
                label_dir=('./xview2/train/labels', './xview2/tier3/labels'),
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
                    ], p=0.75),
                    albu.RandomDiscreteScale([0.75, 1.25, 1.5], p=0.5),
                    RandomCrop(640, 640, True),
                    Normalize(mean=(0.485, 0.456, 0.406,
                                    0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225,
                                   0.229, 0.224, 0.225), max_pixel_value=255),
                    ToTensorV2(True),
                ]),
                batch_size=4,
                num_workers=4,
                training=True
            ),
        ),
        test=dict(
            type='Xview2PairwiseDataLoader',
            params=dict(
                image_dir=('./xview2/train/images', './xview2/tier3/images'),
                label_dir=('./xview2/train/labels', './xview2/tier3/labels'),
                mode='segm',
                include=('pre', 'post'),
                CV=dict(
                    on=True,
                    cur_k=0,
                    k_fold=5,
                ),
                transforms=Compose([
                    Normalize(mean=(0.485, 0.456, 0.406,
                                    0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225,
                                   0.229, 0.224, 0.225), max_pixel_value=255),
                    ToTensorV2(True),
                ]),
                batch_size=1,
                num_workers=0,
                training=False
            ),
        ),
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.0001
        ),
        grad_clip=dict(
            max_norm=35,
            norm_type=2,
        )
    ),
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=0.03,
            power=0.9,
            max_iters=30000,
        )),
    train=dict(
        forward_times=1,
        num_iters=30000,
        eval_per_epoch=False,
        summary_grads=False,
        summary_weights=False,
        distributed=True,
        apex_sync_bn=True,
        sync_bn=False,
        eval_after_train=True,
        log_interval_step=50,
        save_ckpt_interval_epoch=40,
        eval_interval_epoch=40,
    ),
    test=dict(
        tta=[
            ttas.Rotate90k(1),
            ttas.Rotate90k(2),
            ttas.Rotate90k(3),
            ttas.HorizontalFlip(),
            ttas.VerticalFlip(),
            ttas.Transpose(),
            ttas.Scale(scale_factor=0.75),
            ttas.Scale(scale_factor=1.0),
            ttas.Scale(scale_factor=1.25),
            ttas.Scale(scale_factor=1.5),
        ]
    ),
)
