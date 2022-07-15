#!/usr/bin/env python
# coding=UTF-8

import os, sys
import numpy as np
import cv2
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

# 1.修改后的配置文件
config_file = "configs/e2e_ms_rcnn_R_50_FPN_1x.yaml"

# 2.配置
cfg.merge_from_file(config_file) # merge配置文件
cfg.merge_from_list(["MODEL.MASK_ON", True]) # 打开mask开关
cfg.merge_from_list(["MODEL.DEVICE", "cuda"]) # or设置为CPU ["MODEL.DEVICE", "cpu"]
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.1, # 3.设置置信度
)

if __name__ == '__main__':

    in_folder = '/content/drive/MyDrive/maskscoring_rcnn/inference_input/'
    out_folder = '/content/drive/MyDrive/maskscoring_rcnn/inference_output/'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for file_name in os.listdir(in_folder):
        if not file_name.endswith(('jpg', 'png')):
            continue

        # load file
        img_path = os.path.join(in_folder, file_name)
        image = cv2.imread(img_path)

        # method1. 直接得到opencv图片结果
        #predictions = coco_demo.run_on_opencv_image(image)
        #save_path = os.path.join(out_folder, file_name)
        #cv2.imwrite(save_path, predictions)

        # method2. 获取预测结果
        predictions = coco_demo.compute_prediction(image)
        top_predictions = coco_demo.select_top_predictions(predictions)

        # draw
        img = coco_demo.overlay_boxes(image, top_predictions)
        img = coco_demo.overlay_mask(img, predictions)
        img = coco_demo.overlay_class_names(img, top_predictions)
        save_path = os.path.join(out_folder, file_name)
        cv2.imwrite(save_path, img)

        # print results
        boxes = top_predictions.bbox.numpy()
        labels = top_predictions.get_field("labels").numpy()  #label = labelList[np.argmax(scores)]
        scores = top_predictions.get_field("scores").numpy()
        masks = []
        masks = top_predictions.get_field("mask").numpy()

        for i in range(len(boxes)):
            print('box:', i, ' label:', labels[i])
            x1,y1,x2,y2 = [round(x) for x in boxes[i]] # = map(int, boxes[i])
            print('x1,y1,x2,y2:', x1,y1,x2,y2)
