# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 19:06:54 2022

@author: jp341
"""
import json
import xml.dom.minidom
import os

def listFiles():
    fileDir = "C://Users/jp341/Desktop/val_xml/"
    fileList = []
    for root, dirs, files in os.walk(fileDir):
        for fileObj in files:
            fileList.append(os.path.join(root, fileObj))

    for fileObj in fileList:
        f = open(fileObj,'r+')
        all_the_lines = f.readlines()
        f.seek(0)
        f.truncate()
        for line in all_the_lines:
            f.write(line.replace('<?xml version="1.0" encoding="utf-8"?>', ''))
            #print(line)
        f.close()
        
        
with open(r"C:\Users\jp341\Desktop\instances_ships_train2018_small.json","r") as load_f:
    load_dict = json.load(load_f)
    annotations = load_dict["annotations"]
    img = load_dict["images"]
    cate = load_dict["categories"]
    info = load_dict["info"]
    lice = load_dict["licenses"]
    
for i in range(0, len(img)):
    match_ann = []
    for j in range(0, len(annotations)):
        if img[i]["id"] == annotations[j]["image_id"]:
            match_ann.append(annotations[j])
    
    doc = xml.dom.minidom.Document()

    root = doc.createElement("annotation")
    
    folder = doc.createElement("folder")
    folder.appendChild(doc.createTextNode(str('VOC2007')))
    root.appendChild(folder)
    
    filename = doc.createElement("filename")
    filename.appendChild(doc.createTextNode(str(img[i]["file_name"])))
    root.appendChild(filename)
    
    size = doc.createElement("size")
    width = doc.createElement("width")
    width.appendChild(doc.createTextNode(str(img[i]["width"])))
    height = doc.createElement("height")
    height.appendChild(doc.createTextNode(str(img[i]["height"])))
    depth = doc.createElement("depth")
    depth.appendChild(doc.createTextNode(str("3")))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    root.appendChild(size)
    
    segmented = doc.createElement("segmented")
    segmented.appendChild(doc.createTextNode(str('0')))
    root.appendChild(segmented)
    
    for k in range(0, len(match_ann)):
        
        obj = doc.createElement("object")
        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(str("ship")))
        
        truncated = doc.createElement("truncated")
        truncated.appendChild(doc.createTextNode(str("0")))
        
        
        difficult = doc.createElement("difficult")
        difficult.appendChild(doc.createTextNode(str("0")))
        
        
        bndbox = doc.createElement("bndbox")
        
        xmin = doc.createElement("xmin")
        xmin.appendChild(doc.createTextNode(str(match_ann[k]["bbox"][0])))
        ymin = doc.createElement("ymin")
        ymin.appendChild(doc.createTextNode(str(match_ann[k]["bbox"][1])))
        xmax = doc.createElement("xmax")
        xmax.appendChild(doc.createTextNode(str(match_ann[k]["bbox"][0]+match_ann[k]["bbox"][2])))
        ymax = doc.createElement("ymax")
        ymax.appendChild(doc.createTextNode(str(match_ann[k]["bbox"][1]+match_ann[k]["bbox"][3])))
        
        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)
        
        obj.appendChild(name)
        obj.appendChild(truncated)
        obj.appendChild(difficult)
        obj.appendChild(bndbox)
        root.appendChild(obj)
    doc.appendChild(root) 
    
    dirfile = 'C:/Users/jp341/Desktop/val_xml/'+img[i]["file_name"][:-4]+'.xml'
    with open(dirfile, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8')[39:])
    
        