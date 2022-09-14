# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 19:01:54 2022

@author: jp341
"""

import json

with open(r"C:\Users\jp341\Desktop\instances_ships_train2018.json","r") as load_f:
     load_dict = json.load(load_f)
     remain_ann = []
     remain_img = []
     annotations = load_dict["annotations"]
     img = load_dict["images"]
     cate = load_dict["categories"]
     info = load_dict["info"]
     lice = load_dict["licenses"]
     flag = 0
     for i in range(0, len(annotations)):
         
         judge_area = annotations[i]["area"]
         if judge_area>=96*96:
             remain_id = annotations[i]["image_id"]
             remain_ann.append(annotations[i])
             for j in range(0, len(img)):
                 if img[j]["id"] == remain_id:
                     remain_img.append(img[j])
             if len(remain_img)>=800:
                flag = 1
         if flag == 1:
            break;
            
     flag = 0
     for i in range(0, len(annotations)):
         
         judge_area = annotations[i]["area"]
         if judge_area<=32*32:
             remain_id = annotations[i]["image_id"]
             remain_ann.append(annotations[i])
             for j in range(0, len(img)):
                 if img[j]["id"] == remain_id:
                     remain_img.append(img[j])
             if len(remain_img)>=900:
                flag = 1
         if flag == 1:
            break;
     
     flag = 0
     for i in range(0, len(annotations)):
         
         judge_area = annotations[i]["area"]
         if 32*32<judge_area<96*96:
             remain_id = annotations[i]["image_id"]
             remain_ann.append(annotations[i])
             for j in range(0, len(img)):
                 if img[j]["id"] == remain_id:
                     remain_img.append(img[j])
             if len(remain_img)>=1000:
                flag = 1
         if flag == 1:
            break;
     remain_dict = {"info":info, "licenses":lice, "categories":cate, "images":remain_img, "annotations":remain_ann}
     
redict_path = r'C:\Users\jp341\Desktop\instances_ships_train2018_sub811.json'

with open(redict_path,"w") as f:
    json_remain_dict = json.dumps(remain_dict)
    f.write(json_remain_dict)