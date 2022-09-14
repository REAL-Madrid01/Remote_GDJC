from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
from PIL import Image
import torch
import cv2
import scipy.misc
import matplotlib.pyplot as plt
 
class ResNet(nn.Module):
    def __init__(self, num_classes=2):   # num_classes，此处为 二分类值为2
        super(ResNet, self).__init__()
        net = models.resnet50(pretrained=True)   # 从预训练模型加载VGG16网络参数
        pretrained_dict = torch.load(r"C:\Users\jp341\Desktop\resnet50_10kscar_1k.pth", map_location=torch.device('cpu'))
        model_dict = net.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        self.feature = nn.Sequential(*list(net.children())[:-2])
 
    def forward(self, x):
        x = self.feature(x)
        return x
 
def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):#model的第一个Sequential()是有多层，所以遍历
            x = layer(x)#torch.Size([1, 64, 55, 55])生成了64个通道
            if k == index:  
                return x
 
def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')#是一幅图片
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)#torch.Size([3, 224, 224])
    image_info = image_info.unsqueeze(0)#torch.Size([1, 3, 224, 224])因为model的输入要求是4维，所以变成4维
    return image_info#变成tensor数据
 
 
#  可视化特征图
def show_feature_map(feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.detach().numpy().squeeze()  # 压缩成torch.Size([64, 55, 55])
    feature_map_num = feature_map.shape[0]  # 返回通道数
 
    for index in range(feature_map_num):  # 通过遍历的方式，将64个通道的tensor拿出
        feature=feature_map[index]
        feature = np.asarray(feature* 255, dtype=np.uint8)
        #feature=cv2.resize(feature,(224,224),interpolation =  cv2.INTER_NEAREST) #改变特征呢图尺寸
        feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET) #变成伪彩图
        cv2.imwrite('C:/Users/jp341/Desktop/fea/channel_{}.png'.format(str(index)), feature)
 
a = torch.zeros(1,2,2)
b = a
print(b)
#1.将输入图片转成tensor的形式
image_info=get_image_info(r"D:\Git\Data Store\all\all\airbus\ships_train2018_small\41f68e685.jpg")
 
#2.指定可视化第一层feature,并且将特征图取出
my_model=ResNet()
model_layer= list(my_model.children())
model_layer=model_layer[0]#这里选择model的第一个Sequential()
feature_map = get_k_layer_feature_map(model_layer, 0, image_info)
 
#3.将特征图逐层保存
show_feature_map(feature_map)