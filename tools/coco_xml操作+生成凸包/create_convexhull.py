# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:47:47 2022

@author: jp341
"""

import torch,torchvision
import numpy as np
from PIL import Image
import cv2
import math
"""
    随机选取defect{m_i}中的一张图像resize到背景non-defect{b_i}大小并全零化
"""
def prepare_backgound(nondefect_filepath):
    
    
    b_i = Image.open(nondefect_filepath)
    w = b_i.width
    h = b_i.height
    
    m_i = np.zeros((h, w, 3), dtype='uint8')
    m_i = Image.fromarray(m_i)
    return b_i, m_i
    
def calculate_area(img):
    img_array = torch.tensor(np.asarray(img).astype(float))
    area = (img_array.reshape(-1, 3)[:,1] == 255).sum()
    print(area)
    return area

def coordinate_ellipse(a, b, thet):
    tan_thet = math.tan(math.radians(thet))
    x = (a*b)/math.sqrt(b*b+math.pow(a*tan_thet, 2))
    y = (a*b*tan_thet)/math.sqrt(b*b+math.pow(a*tan_thet, 2))
    if 0<=thet<90 or 270<=thet<360:
       x = x
       y = y
    else:
       x = -x
       y = -y
    return np.array([x, y]).T

def create_convexhull(background_img, Rl, Rw):
    
    background_img = np.asarray(background_img).astype(float)
    center = np.array([(Rl[0]+Rl[1])/2.0, (Rw[0]+Rw[1])/2.0]).astype(int)
    axes_l = Rl[1]-Rl[0]
    axes_w = Rw[1]-Rw[0]
    angle = np.random.randint(0,15)
    
    cos_angle = math.cos(math.radians(angle))
    sin_angle = math.sin(math.radians(angle))
    
    R_T = np.array([[cos_angle,-sin_angle],
                    [sin_angle, cos_angle]]).astype(float)
    
    center_T = center.T
    a = axes_l/2.0
    b = axes_w/2.0
    """
        axis (X,Y) from left to right from top to bottle
        begin 'fillpoly' at left and clockwise 
    """
    xy_T_real = []
    
    
    xy_T = coordinate_ellipse(a, b, 179)
    xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
    xy_T = coordinate_ellipse(a, b, 180)
    xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
    ellipse_u = 180
    for i in range(1, 3):
        incre_thet = np.random.randint(10, 30)
        ellipse_u += incre_thet
        xy_T = coordinate_ellipse(a, b, ellipse_u)
        xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
        xy_T = coordinate_ellipse(a, b, ellipse_u+1)
        xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
        xy_T = coordinate_ellipse(a, b, ellipse_u+2)
        xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
        
    xy_T = coordinate_ellipse(a, b, 359)
    xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
    xy_T = coordinate_ellipse(a, b, 0)
    xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
    xy_T = coordinate_ellipse(a, b, 1)
    xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
    xy_T = coordinate_ellipse(a, b, 2)
    xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
    xy_T = coordinate_ellipse(a, b, 3)
    xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
    ellipse_d = 0
    # for i in range(1, 3):
    #     incre_thet = np.random.randint(20, 40)
    #     ellipse_d += incre_thet
    #     xy_T = coordinate_ellipse(a, b, ellipse_d)
    #     xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
    #     xy_T = coordinate_ellipse(a, b, ellipse_d+1)
    #     xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
    #     xy_T = coordinate_ellipse(a, b, ellipse_d+2)
    #     xy_T_real.append((R_T.dot(xy_T)+center_T).astype(int))
    
    print(center)
    print([a, b])
    print([[center[0]-a, center[1]],[center[0], center[1]-b],
           [center[0]+a, center[1]],[center[0], center[1]+b]])
    # cv2.ellipse(background_img, center, (axes_l, axes_w), angle,  360, 0, (255, 255, 255), 3)
    
    background_img = cv2.fillPoly(background_img, [np.array(xy_T_real)], [255, 255, 255])
    background_img  = Image.fromarray(background_img.astype('uint8')).convert('RGB')
    area = calculate_area(background_img)
    # background_img.show()
    
    return background_img, xy_T_real, area


def create_defect(background_img=None, Smin=0, mode=None):
    
    
    assert background_img != None ,"img is empty!"
    assert Smin > 0 ,"Smin must set as a specfiy num!"
    assert mode in ['slender_ha', 'slender_va', 'circular'], "mode invalid!"
    w = background_img.height
    l = background_img.width
    area = 0
    xy_T_real = []
    if mode=='slender_ha':
        while area < Smin:
            coin_l = np.random.randint(0, 2)
            coin_w = np.random.randint(0, 2)
            if coin_l == 0:
                bg_l = np.random.randint(0.2*l, 0.3*l)
                ed_l = np.random.randint(0.4*l, 0.45*l)
            if coin_l == 1:
                bg_l = np.random.randint(0.5*l, 0.56*l)
                ed_l = np.random.randint(0.7*l, 0.75*l)
            if coin_w == 0:
                bg_w = np.random.randint(0.1*w, 0.5*w)
                ed_w = bg_w+10
            if coin_w == 1:
                bg_w = np.random.randint(0.5*w, 0.9*w)
                ed_w = bg_w+10
            Rl = [bg_l, ed_l]
            Rw = [bg_w, ed_w]
            
            background_img, xy_T_real_oneshot, area = create_convexhull(background_img, Rl, Rw)
            xy_T_real.append(xy_T_real_oneshot)
            
        return xy_T_real, background_img

bi_path = r'D:\Tencent\WeChat Files\wxid_psoobqh293e122\FileStorage\File\No defects\ships_test2018\3b01e6641.jpg'
or_path = r'C:\Users\jp341\Desktop\train_hq\train_hq\0cdf5b5d0ce1_01.jpg'
b_i, m_i = prepare_backgound(bi_path)
xy_T_real, defectground_img = create_defect(m_i, Smin=1000, mode="slender_ha")
b_i = torch.from_numpy(np.asarray(b_i))
m_i = torch.from_numpy(np.asarray(m_i))
dg_i = torch.tensor(np.asarray(defectground_img))
idx = torch.nonzero(dg_i==255, as_tuple=(True))
dg_i_n = torch.ones(dg_i.size())
dg_i_n[idx] = 0
dg_i_p = torch.zeros(dg_i.size())
dg_i_p[idx] = 1
# defect_img  = Image.fromarray(np.asarray(dg_i_n.numpy()).astype('uint8')).convert('RGB')
# defect_img.show()
b_i_min = torch.min(b_i)
print(b_i_min)
b_i_n = b_i.numpy().astype(float)
d_i_a = (dg_i_p*b_i_min).numpy().astype(float)
d_i = cv2.multiply(b_i.numpy().astype(float), dg_i_n.numpy().astype(float))+(dg_i_p*b_i_min).numpy().astype(float)
defect_img = Image.fromarray(d_i.astype('uint8')).convert('RGB')
defect_img.show()