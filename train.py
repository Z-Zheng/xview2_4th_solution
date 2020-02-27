from simplecv import apex_ddp_train as train
from data import xview2_loader
from module import register_model

import torch
import simplecv as sc
import time
from simplecv.util.logger import eval_progress, speed
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from simplecv.core.transform_base import ParallelTestTransform


def tta(model, image, tta_config):
    trans = ParallelTestTransform(
        *tta_config
    )
    images = trans.transform(image)
    with torch.no_grad():
        outs = [model(im) for im in images]

    outs = trans.inv_transform(outs)

    out = sum(outs) / len(outs)

    return out


def evaluate_cls_fn(self, test_dataloader, config=None):
    vis_dir = os.path.join(self.model_dir, 'vis-{}'.format(self.checkpoint.global_step))
    torch.cuda.empty_cache()
    self.model.eval()
    total_time = 0.
    ppe = ProcessPoolExecutor(max_workers=4)
    metric_op = sc.metric.NPPixelMertic(max(self.model.module.config.head.num_classes, 2), self.model_dir)
    # 0 - bg, 1 - building
    viz_op = sc.viz.VisualizeSegmm(vis_dir, [0, 0, 0, 0, 0, 255])
    with torch.no_grad():
        for idx, (ret, ret_gt) in enumerate(test_dataloader):
            start = time.time()
            ret = ret.to(torch.device('cuda'))

            if config is not None and 'tta' in config:
                y = tta(self.model, ret, config['tta'])
            else:
                y = self.model(ret)

            cls = (y > 0.5).cpu()
            cls = cls.numpy()
            cls_gt = ret_gt['cls']
            cls_gt = cls_gt.numpy()
            y_true = cls_gt.ravel()
            y_pred = cls.ravel()

            y_true = np.where(y_true > 0, np.ones_like(y_true), np.zeros_like(y_true))

            metric_op.forward(y_true, y_pred)

            time_cost = round(time.time() - start, 3)

            total_time += time_cost

            filename = ret_gt['image_filename']
            ppe.submit(viz_op, cls, filename[0].replace('jpg', 'png'))

            speed(self._logger, time_cost, 'batch')
            eval_progress(self._logger, idx + 1, len(test_dataloader))
    torch.cuda.empty_cache()
    speed(self._logger, round(total_time / len(test_dataloader), 3), 'batch (avg)')

    metric_op.summary_all()
    torch.cuda.empty_cache()


def evaluate_loc_cls_fn(self, test_dataloader, config=None):
    # vis_dir = os.path.join(self.model_dir, 'vis-{}'.format(self.checkpoint.global_step))
    torch.cuda.empty_cache()
    self.model.eval()
    total_time = 0.
    # ppe = ProcessPoolExecutor(max_workers=4)
    loc_metric_op = sc.metric.NPPixelMertic(2, self.model_dir)
    damage_metric_op = sc.metric.NPPixelMertic(max(self.model.module.config.head.num_classes, 2), self.model_dir)
    # # 0 - bg, 1 - building
    # viz_op = sc.viz.VisualizeSegmm(vis_dir, [0, 0, 0, 0, 0, 255])
    with torch.no_grad():
        for idx, (ret, ret_gt) in enumerate(test_dataloader):
            start = time.time()
            ret = ret.to(torch.device('cuda'))

            if config is not None and 'tta' in config:
                y = tta(self.model, ret, config['tta'])
            else:
                y = self.model(ret)
            loc_prob = y[:, :1, :, :]

            loc_pred = (loc_prob > 0.5).cpu().numpy().ravel()

            gt = ret_gt['cls']
            loc_true = gt[:, :, :, 0].numpy().ravel()
            loc_true = np.where(loc_true > 0, np.ones_like(loc_true), np.zeros_like(loc_true))
            loc_metric_op.forward(loc_true, loc_pred)

            dam_prob = y[:, 1:, :, :]
            dam_pred = dam_prob.argmax(dim=1).cpu().numpy().ravel()
            dam_true = gt[:, :, :, 1].numpy().ravel()

            valid_inds = np.where(dam_true != self.model.module.config.loss.ignore_index)[0]
            dam_true = dam_true[valid_inds]
            dam_pred = dam_pred[valid_inds]

            damage_metric_op.forward(dam_true, dam_pred)

            time_cost = round(time.time() - start, 3)
            total_time += time_cost

            speed(self._logger, time_cost, 'batch')
            eval_progress(self._logger, idx + 1, len(test_dataloader))
    torch.cuda.empty_cache()
    speed(self._logger, round(total_time / len(test_dataloader), 3), 'batch (avg)')

    loc_metric_op.summary_all()
    damage_metric_op.summary_all()

    torch.cuda.empty_cache()


def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_cls_fn)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    args = train.parser.parse_args()
    train.run(local_rank=args.local_rank,
              config_path=args.config_path,
              model_dir=args.model_dir,
              opt_level=args.opt_level,
              cpu_mode=args.cpu,
              after_construct_launcher_callbacks=[register_evaluate_fn],
              opts=args.opts)
