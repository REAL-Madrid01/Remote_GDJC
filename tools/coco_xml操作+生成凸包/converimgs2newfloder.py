# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:49:38 2022

@author: jp341
"""
import json
from PIL import Image

with open(r"C:\Users\jp341\Desktop\instances_ships_train2018_small.json","r") as load_f:
    load_dict = json.load(load_f)
    img = load_dict["images"]
    file_name = []
    for i in range(0, len(img)):
        file_name.append(img[i]["file_name"])
        

dirname_read = r"D://Git/Data Store/all/all/airbus/ships_train2018/"
dirname_write = r"C:\Users\jp341\Desktop\val_img/"
for i in range(0, len(file_name)):
    img_name = dirname_read+file_name[i]
    img = Image.open(img_name)
    img.save(dirname_write+file_name[i])
    