from __future__ import division

import numpy as np

import chainer
import chainer.functions as F


def focal_loss(x, t, gamma=2, alpha=0.25):
    xp = chainer.cuda.get_array_module(x)
    t = xp.eye(x.shape[1])[t]
    return -alpha * F.mean(F.matmul((1 - F.softmax(x))**gamma * F.log_softmax(x), t, transb=True))


def multibox_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k):
    """Computes multibox losses.
    This is a loss function used in [#]_.
    This function returns :obj:`loc_loss` and :obj:`conf_loss`.
    :obj:`loc_loss` is a loss for localization and
    :obj:`conf_loss` is a loss for classification.
    The formulas of these losses can be found in
    the equation (2) and (3) in the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        mb_locs (chainer.Variable or array): The offsets and scales
            for predicted bounding boxes.
            Its shape is :math:`(B, K, 4)`,
            where :math:`B` is the number of samples in the batch and
            :math:`K` is the number of default bounding boxes.
        mb_confs (chainer.Variable or array): The classes of predicted
            bounding boxes.
            Its shape is :math:`(B, K, n\_class)`.
            This function assumes the first class is background (negative).
        gt_mb_locs (chainer.Variable or array): The offsets and scales
            for ground truth bounding boxes.
            Its shape is :math:`(B, K, 4)`.
        gt_mb_labels (chainer.Variable or array): The classes of ground truth
            bounding boxes.
            Its shape is :math:`(B, K)`.
        k (float): A coefficient which is used for hard negative mining.
            This value determines the ratio between the number of positives
            and that of mined negatives. The value used in the original paper
            is :obj:`3`.
    Returns:
        tuple of chainer.Variable:
        This function returns two :obj:`chainer.Variable`: :obj:`loc_loss` and
        :obj:`conf_loss`.
    """
    mb_locs = chainer.as_variable(mb_locs)
    mb_confs = chainer.as_variable(mb_confs)
    gt_mb_locs = chainer.as_variable(gt_mb_locs)
    gt_mb_labels = chainer.as_variable(gt_mb_labels)

    xp = chainer.cuda.get_array_module(gt_mb_labels.array)

    positive = gt_mb_labels.array > 0
    n_positive = positive.sum()
    if n_positive == 0:
        z = chainer.Variable(xp.zeros((), dtype=np.float32))
        return z, z

    loc_loss = F.huber_loss(mb_locs, gt_mb_locs, 1, reduce='no')
    loc_loss = F.sum(loc_loss, axis=-1)
    loc_loss *= positive.astype(loc_loss.dtype)
    loc_loss = F.sum(loc_loss) / n_positive

    conf_loss = focal_loss(mb_confs, gt_mb_labels)

    return loc_loss, conf_loss
