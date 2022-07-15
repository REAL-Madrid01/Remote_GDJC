# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import numpy as np
import cv2
from torchvision.utils import make_grid

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)


    #  可视化特征图
    def show_feature_map(self, feature_map=None, f=None):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
      # feature_map[2].shape     out of bounds
      feature_map = feature_map.detach().cpu().numpy().squeeze()  # 压缩成torch.Size([64, 55, 55])
      feature_map_num = feature_map.shape[0]  # 返回通道数
 
      for index in range(feature_map_num):  # 通过遍历的方式，将64个通道的tensor拿出
        feature=feature_map[index]
        feature = np.asarray(feature* 1, dtype=np.uint8)
        #feature=cv2.resize(feature,(224,224),interpolation =  cv2.INTER_NEAREST) #改变特征呢图尺寸
        feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET) #变成伪彩图
        cv2.imwrite('/content/drive/MyDrive/maskscoring_rcnn/inference_output/fea'+str(f)+'/channel_{}.png'.format(str(index)), feature)


    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        """
          对于单独的ResNet，取出features[1]作为features
        """
        features_revise = []
        #feature_empty = torch.empty(2, 256, 25, 38)
        features_revise.append(features[0])
        features_revise.append(features[1])
        # print("****************特征图个数{}".format(len(features)))
        # print("****************特征图尺寸{}".format(type(features[0])))
        # print("****************特征图尺寸{}".format(features[1].size()))
        # print("****************特征图尺寸{}".format(features[2].size()))
        # print("****************特征图尺寸{}".format(features[3].size()))
        # print("****************特征图尺寸{}".format(features[4].size()))
        # self.show_feature_map(feature_map=features[0], f=0)
        # self.show_feature_map(feature_map=features[1], f=1)
        # self.show_feature_map(feature_map=features[2], f=2)
        # self.show_feature_map(feature_map=features[3], f=3)
        # self.show_feature_map(feature_map=features[4], f=4)
        #print("features_revise的类型为{}".format(type(features_revise)))
        #print("features[0]的形状为{}".format(features[0].size()))
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
