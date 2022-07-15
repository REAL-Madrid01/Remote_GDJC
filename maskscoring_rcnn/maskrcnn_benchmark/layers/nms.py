# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#from ._utils import _C
from maskrcnn_benchmark import _C

nms = _C.nms
#nms.__doc__ = """
# This function performs Non-maximum suppresion"""
import torch
import numpy as np

# def non_nms(bboxes, scores, threshold):
#         #print("使用了nms")
#         #print("bboxes为{}".format(bboxes.size()))
#         x1 = bboxes[:,0]
#         #print("x1为{}".format(x1.size()))
#         y1 = bboxes[:,1]
#         x2 = bboxes[:,2]
#         y2 = bboxes[:,3]
#         areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
#         _, order = scores.sort(0, descending=True)    # 降序排列
        
#         keep = []
#         while order.numel() > 0:   # torch.numel()返回张量元素个数
#             if order.numel() == 1:     # 保留框只剩一个
#                 i = order.item()
#                 keep.append(i)
#                 break
#             else:
#                 i = order[0].item()    # 保留scores最大的那个框box[i]
#                 keep.append(i)
                
#         return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor


# def nms_origin(bboxes, scores, threshold):
#         #print("使用了nms")
#         #print("bboxes为{}".format(bboxes.size()))
#         x1 = bboxes[:,0]
#         #print("x1为{}".format(x1.size()))
#         y1 = bboxes[:,1]
#         x2 = bboxes[:,2]
#         y2 = bboxes[:,3]
#         areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
#         _, order = scores.sort(0, descending=True)    # 降序排列
        
#         keep = []
#         while order.numel() > 0:   # torch.numel()返回张量元素个数
#             if order.numel() == 1:     # 保留框只剩一个
#                 i = order.item()
#                 keep.append(i)
#                 break
#             else:
#                 i = order[0].item()    # 保留scores最大的那个框box[i]
#                 keep.append(i)

#             # 计算box[i]与其余各框的IOU(思路很好)
#             xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
#             yy1 = y1[order[1:]].clamp(min=y1[i])
#             xx2 = x2[order[1:]].clamp(max=x2[i])
#             yy2 = y2[order[1:]].clamp(max=y2[i])
#             inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]
            
#             #print()
#             #g = np.nonzero(inter)
#             #print(g)
#             #for j in range(0, len(inter)):
#             #  if int(areas[i])==int(inter.tolist()[j]):
#             #    inter.tolist()[j] = 1000*inter.tolist()[j]

#             iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
            
#             # g = (inter == areas[i]).nonzero()
#             # if(len(g) > 0):
#             #   g = g.squeeze()
#             #   #remove是保留最大框，iou[0] = 1是保留小框
#             #   #iou[0] = 1  
#             #   keep.remove(i)
#             idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
#             if idx.numel() == 0:
#                 break
#             #print(order.size(), order[idx+1].size())
#             order = order[idx+1]  # 修补索引之间的差值
#         return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor