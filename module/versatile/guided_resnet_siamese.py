from module.versatile.base import GeneralizedSegm
import torch.nn as nn
from simplecv import registry
from simplecv.module.resnet import ResNetEncoder
from simplecv.module.fpn import FPN, conv_bn_block
import torch.nn.functional as F
from simplecv.module import loss
import torch
import numpy as np
from simplecv.module import se_block
from module.versatile.loss import ohem, tversky_loss_with_logits
import math


class GuidedAttention(nn.Module):
    def __init__(self, inchannels):
        super(GuidedAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(True)
        )

    def forward(self, loc_logit, fused_feat):
        loc_prob = loc_logit.sigmoid()
        indentity = fused_feat
        fused_feat = self.conv(fused_feat)
        fused_feat = fused_feat * loc_prob + indentity
        return fused_feat


class FuseConv(nn.Sequential):
    def __init__(self, inchannels, outchannels):
        super(FuseConv, self).__init__(nn.Conv2d(inchannels, outchannels, kernel_size=1),
                                       nn.BatchNorm2d(outchannels),
                                       )
        self.relu = nn.ReLU(True)
        self.se = se_block.SEBlock(outchannels, 16)

    def forward(self, x):
        out = super(FuseConv, self).forward(x)
        residual = out
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class AssymetricDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super(AssymetricDecoder, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        return out_feat


class DeepHead(nn.Module):
    def __init__(self, in_channels, bottlneck_channels, num_blocks, num_classes, upsample_scale):
        super(DeepHead, self).__init__()
        assert num_blocks > 0
        self.relu = nn.ReLU(True)
        self.blocks = nn.ModuleList([nn.Sequential(
            # 1x1
            nn.Conv2d(in_channels, bottlneck_channels, 1),
            nn.BatchNorm2d(bottlneck_channels),
            nn.ReLU(True),
            # 3x3
            nn.Conv2d(bottlneck_channels, bottlneck_channels, 3, 1, 1),
            nn.BatchNorm2d(bottlneck_channels),
            # 1x1
            nn.Conv2d(bottlneck_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            se_block.SEBlock(in_channels, 16)
        ) for _ in range(num_blocks)])

        self.cls = nn.Conv2d(in_channels, num_classes, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)

    def forward(self, x, upsample=True):
        indentity = x
        for m in self.blocks:
            x = m(x)
            x += indentity
            x = self.relu(x)
            indentity = x
        x = self.cls(x)
        if upsample:
            x = self.up(x)
        return x


@registry.MODEL.register('GSiameseResNet')
class SiameseResNet(GeneralizedSegm):
    def __init__(self, config):
        super(SiameseResNet, self).__init__(config)
        out_channels = self.config.neck.out_channels
        self.fuse_conv = FuseConv(2 * out_channels, out_channels)

        neck_cfg = self.config.neck
        head_cfg = self.config.head
        self.loc_neck = nn.Sequential(
            FPN(neck_cfg.in_channels_list, neck_cfg.out_channels, conv_bn_block),
            AssymetricDecoder(head_cfg.in_channels, head_cfg.out_channels)
        )

        self.dam_neck = nn.Sequential(
            FPN(neck_cfg.in_channels_list, neck_cfg.out_channels, conv_bn_block),
            AssymetricDecoder(head_cfg.in_channels, head_cfg.out_channels)
        )

        self.loc_head = DeepHead(self.config.head.in_channels,
                                 self.config.head.bottleneck_channels,
                                 self.config.head.num_blocks,
                                 1,
                                 self.config.head.upsample_scale)

        self.damage_head = DeepHead(self.config.head.in_channels,
                                    self.config.head.bottleneck_channels,
                                    self.config.head.num_blocks,
                                    self.config.head.num_classes,
                                    self.config.head.upsample_scale)

        self.gatt = GuidedAttention(self.config.guided_att.in_channels)
        self.up = nn.UpsamplingBilinear2d(scale_factor=self.config.head.upsample_scale)

    def forward(self, x, y=None):
        """

        Args:
            x: [N, 2*3, H, W]
            y: dict(cls=mask), mask is of shape [N, H, W, 2]

        Returns:

        """
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        # loc
        feat1 = self.loc_neck(self.ops.backbone(x1))
        logit1 = self.loc_head(feat1, upsample=False)
        # siamese dam
        feat2 = self.dam_neck(self.ops.backbone(x2))

        fuse_feat = self.fuse_conv(torch.cat([feat1, feat2], dim=1))

        fuse_feat = self.gatt(logit1, fuse_feat)
        logit1 = self.up(logit1)
        logit2 = self.damage_head(fuse_feat)

        if self.training:
            y_true = y['cls']
            pre_y_true = y_true[:, :, :, 0]
            post_y_true = y_true[:, :, :, 1]
            loss_dict = dict()
            loss_dict.update(self.loc_loss(logit1, pre_y_true))
            loss_dict.update(self.damage_loss(logit2, post_y_true))

            # iou-1
            with torch.no_grad():
                y_pred = (logit1.sigmoid() > 0.5).float().view(-1)
                y_true = pre_y_true.float().view(-1)
                inter = torch.sum(y_pred * y_true)
                union = y_true.sum() + y_pred.sum()
                loss_dict['iou-1'] = inter / torch.max(union - inter, torch.as_tensor(1e-6, device=y_pred.device))

                y_pred = logit2.argmax(dim=1)
                dam_acc = (y_pred == post_y_true.long()).float().sum() / post_y_true.numel()
                loss_dict['dam_acc'] = dam_acc

            mem = torch.cuda.max_memory_cached() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict

        return torch.cat([logit1.sigmoid(), logit2.softmax(dim=1)], dim=1)

    def backbone(self):
        module = ResNetEncoder(self.config.backbone)
        return module

    def neck(self):
        return nn.Identity()

    def head(self):
        return nn.Identity()

    def loc_loss(self, y_pred, y_true):
        loss_dict = dict()
        if 'dice_loss' in self.config.loss.loc:
            loss_dict.update(dict(
                dice_loss=loss.dice_loss_with_logits(y_pred, y_true),
            ))
        if 'bce_loss' in self.config.loss.loc:
            if 'ohem' in self.config.loss.loc:
                ratio = self.config.loss.loc.ohem.ratio
                bce_losses = F.binary_cross_entropy_with_logits(y_pred.view(-1),
                                                                y_true.float().view(
                                                                    -1), reduction='none')
                loss_dict.update(dict(ohem_bce_loss=ohem(bce_losses, ratio)))
            else:
                loss_dict.update(dict(
                    bce_loss=F.binary_cross_entropy_with_logits(y_pred.view(-1),
                                                                y_true.float().view(
                                                                    -1))
                ))
        if 'tversky_loss' in self.config.loss.loc:
            _cfg = self.config.loss.loc.tversky_loss
            loss_dict.update(dict(
                tversky_loss=tversky_loss_with_logits(y_pred, y_true, _cfg.alpha, _cfg.beta)
            ))
        return loss_dict

    def damage_loss(self, y_pred, y_true):
        loss_dict = dict()
        if 'ohem' in self.config.loss.dam:
            ratio = self.config.loss.dam.ohem.ratio
            cls_losses = F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index,
                                         reduction='none')
            loss_dict.update(dict(
                ohem_cls_loss=ohem(cls_losses, ratio)
            ))
        else:
            loss_dict.update(dict(
                cls_loss=F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)
            ))
        return loss_dict

    def set_defalut_config(self):
        super(SiameseResNet, self).set_defalut_config()
        self.config.pop('resnet_encoder')
        self.config.update(dict(
            backbone=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=False,
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
                bottleneck_channels=256
            ),
            guided_att=dict(
                in_channels=256,
            ),
            loss=dict(
                ignore_index=255,
                loc=dict(
                )
            ),
        ))

