import torch
import torch.nn as nn
from simplecv.interface import CVModule
from simplecv.module.resnet import ResNetEncoder
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from simplecv.module import loss


class GeneralizedSegm(CVModule):
    def __init__(self, config):
        super(GeneralizedSegm, self).__init__(config)
        self.ops = nn.Sequential(OrderedDict([
            ('backbone', self.backbone()),
            ('neck', self.neck()),
            ('head', self.head()),
        ]))
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def backbone(self):
        return ResNetEncoder(self.config.resnet_encoder)

    def neck(self):
        return nn.Identity()

    def head(self):
        cfg = self.config.head
        return nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=cfg.upsample_scale),
        )

    def forward(self, x, y=None):
        logit = self.ops(x)
        if self.training:
            cls_true = y['cls']
            binary_cls_true = torch.where(cls_true > 0, torch.ones_like(cls_true), torch.zeros_like(cls_true))
            loss_dict = self.cls_loss(logit, binary_cls_true)

            mem = torch.cuda.max_memory_cached() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)

            # iou-1
            with torch.no_grad():
                y_pred = (logit.sigmoid() > 0.5).float().view(-1)
                y_true = binary_cls_true.float().view(-1)
                inter = torch.sum(y_pred * y_true)
                union = y_true.sum() + y_pred.sum()
                loss_dict['iou-1'] = inter / torch.max(union - inter, torch.as_tensor(1e-6, device=y_pred.device))

            return loss_dict
        return logit

    def cls_loss(self, y_pred, y_true):
        loss_dict = dict(dice_loss=loss.dice_loss_with_logits(y_pred, y_true),
                         bce_loss=F.binary_cross_entropy_with_logits(y_pred.view(-1),
                                                                     y_true.float().view(
                                                                         -1)))
        return loss_dict

    def set_defalut_config(self):
        self.config.update(dict(
            resnet_encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=False,
                freeze_at=0,
                # 8, 16 or 32
                output_stride=16,
                with_cp=(False, False, False, False),
                stem3_3x3=False,
            ),
            neck=dict(
            ),
            head=dict(
                in_channels=256,
                num_classes=1,
                upsample_scale=16.0,
            ),
            loss=dict(
                ignore_index=255,
            )
        ))

    def assert_shape(self, shape=(1, 3, 512, 512)):
        self.eval()
        out = self(torch.ones(*shape).to(self.device))
        assert shape[2] == out.size(2) and shape[3] == out.size(3)
        print(out.shape)
