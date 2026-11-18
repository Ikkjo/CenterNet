# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchvision.ops import nms as torchvision_nms

from model.config import cfg
from nms.cpu_nms import cpu_nms

def nms(dets, thresh, force_cpu=False):
  """Dispatch to either CPU or GPU NMS implementations."""

  if dets.shape[0] == 0:
    return []
  if cfg.USE_GPU_NMS and not force_cpu and torch.cuda.is_available():
    boxes = torch.as_tensor(dets[:, :4], device="cuda").clone()
    # VOC uses inclusive pixel coordinates, whereas torchvision uses continuous boxes.
    boxes[:, 2:] += 1
    scores = torch.as_tensor(dets[:, 4], device="cuda")
    return torchvision_nms(boxes, scores, thresh).cpu().tolist()
  return cpu_nms(dets, thresh)
