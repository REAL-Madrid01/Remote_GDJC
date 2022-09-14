# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:52:00 2022

@author: jp341
"""

import json
from PIL import Image

with open(r"C:\Users\jp341\Desktop\instances_ships_train2018_small.json","r") as load_f:
    load_dict = json.load(load_f)
    img = load_dict["images"]
    file_name = []
    for i in range(0, len(img)):
        file_name.append(img[i]["file_name"][:-4])


f = open(r"C:\Users\jp341\Desktop\val.txt", "w")
for i in range(0, len(file_name)):
    print(i)
    f.write(file_name[i])
    f.write('\n')