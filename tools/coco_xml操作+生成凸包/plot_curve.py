# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:47:20 2022

@author: jp341
"""

import numpy as np
import matplotlib.pyplot as plt  
x1=[50,100,150,200,250,300,350,400,450,500,550]
y1=[0.835,0.545,0.491,0.398,0.369,0.337,0.316,0.287,0.274,0.25,0.233]
x2=x1
y2=[1.701,1.238,1.118,1.097,1.208,1.036,0.919,0.882,0.894,0.862,0.812]

#x=np.arange(50,550)

l1=plt.plot(x1,y1,'r--',label='Reg.Loss')
l2=plt.plot(x2,y2,'g--',label='Cls.Loss')
plt.plot(x1,y1,'ro-',x2,y2,'go-')
plt.xlabel('epochs')
plt.ylabel('average loss')
plt.legend()
plt.show()

# x1=[50,100,150,200,250,300,350,400,450,500,550]
# y1=[0.025,0.013,0.038,0.029,0.011,0.006,0.014,0.094,0.01,0.009,0.018]

# #x=np.arange(50,550)
# plt.ylim(0, 0.3);
# l1=plt.plot(x1,y1,'r--',label='mAP')
# plt.plot(x1,y1,'ro-')
# plt.xlabel('epochs')
# plt.ylabel('mAP')
# plt.legend()
# plt.show()
