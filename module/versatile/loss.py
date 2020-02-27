import torch


def ohem(loss, ratio):
    # 1. keep num
    num_inst = loss.numel()
    num_hns = int(ratio * num_inst)
    # 2. select loss
    top_loss, _ = loss.reshape(-1).topk(num_hns, -1)
    loss_mask = (top_loss != 0)
    # 3. mean loss
    return torch.sum(top_loss[loss_mask]) / loss_mask.sum()


@torch.jit.script
def tversky_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float, beta: float,
                             smooth_value: float = 1.0,
                             ignore_index: int = 255):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    mask = y_true == ignore_index
    valid = 1 - mask
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()

    y_pred = y_pred.sigmoid()
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    tversky_coeff = (tp + smooth_value) / (tp + alpha * fn + beta * fp + smooth_value)
    return 1. - tversky_coeff
