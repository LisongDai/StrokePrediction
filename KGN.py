# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import numpy as np



class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)
    
    
class predictor(nn.Module):
    def __init__(self, in_channels, num_class, scale, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(predictor, self).__init__()
        self.conv1 = ConvBnLeakyRelu2d(in_channels, in_channels, kernel_size = 1, padding=0)
        self.conv2 = ConvBnLeakyRelu2d(in_channels, in_channels // 2, kernel_size = 1, padding=0)
        self.conv3 = ConvBnLeakyRelu2d(in_channels // 2, num_class, kernel_size = 3)
        self.scale = scale
        self.act   = nn.ReLU(True) 
        
    def forward(self, x):
        
        x = self.conv1(x) + x
        x = self.conv2(x)
        
        return self.conv3(x) # self.act(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p


class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)    



class KGN(nn.Module):

    def __init__(self, n_classes):
        super(KGN, self).__init__()
        
        model1 = models.resnet18(pretrained=True) # 18 34 50 152
        model2 = models.resnet18(pretrained=True) # 18 34 50 152

        # Image encoding via resnet 
        self.encoder_RGB3_conv1 = model1.conv1
        self.encoder_RGB3_bn1 = model1.bn1
        self.encoder_RGB3_relu = model1.relu
        self.encoder_RGB3_maxpool = model1.maxpool
        self.encoder_RGB3_layer1 = model1.layer1
        self.encoder_RGB3_layer2 = model1.layer2
        self.encoder_RGB3_layer3 = model1.layer3
        self.encoder_RGB3_layer4 = model1.layer4
        
        # knowledge encoding via resnet
        self.encoder_T3_conv1 = model2.conv1
        self.encoder_T3_bn1 = model2.bn1
        self.encoder_T3_relu = model2.relu
        self.encoder_T3_maxpool = model2.maxpool
        self.encoder_T3_layer1 = model2.layer1
        self.encoder_T3_layer2 = model2.layer2
        self.encoder_T3_layer3 = model2.layer3
        self.encoder_T3_layer4 = model2.layer4
        

        self.C0 = nn.Conv2d(in_channels=512*1, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.C1 = nn.Conv2d(in_channels=512*1, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.C2 = nn.Conv2d(in_channels=256*1, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.C3 = nn.Conv2d(in_channels=256*1, out_channels=128, kernel_size=1, stride=1, padding=0)
        
        self.P0 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.P1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.P2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.predictor    = predictor(32, n_classes, 2)
        
    def forward(self, mri, kn):

        x_rgb = mri
        x_inf = kn

        # encode
        rgb = self.encoder_RGB3_conv1(x_rgb)
        rgb = self.encoder_RGB3_bn1(rgb)
        rgb = self.encoder_RGB3_relu(rgb)
        rgb = self.encoder_RGB3_maxpool(rgb)           
        rgb = self.encoder_RGB3_layer1(rgb)  
        tt = self.encoder_T3_conv1(x_inf)
        tt = self.encoder_T3_bn1(tt)
        tt = self.encoder_T3_relu(tt)
        tt = self.encoder_T3_maxpool(tt)
        tt = self.encoder_T3_layer1(tt)       
        rgb = self.encoder_RGB3_layer2(rgb)
        tt = self.encoder_T3_layer2(tt)    
        rgb = self.encoder_RGB3_layer3(rgb)
        tt = self.encoder_T3_layer3(tt)      
        rgb = self.encoder_RGB3_layer4(rgb)   
        tt = self.encoder_T3_layer4(tt)
 
        # feature fusion
        fu = self.C0(rgb) + self.C1(tt)
        fu = F.upsample(fu, scale_factor=2, mode='nearest') 
        fu = self.C2(fu) + fu
        fu = F.upsample(fu, scale_factor=2, mode='nearest') 
        fu = self.C3(fu)
        
        # prediction
        
        x = self.P0(fu)
        x = F.upsample(x, scale_factor=2, mode='nearest') 
        x = self.P1(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') 
        x = self.P2(x) + x
        semantic = self.predictor(x) 
        semantic = F.upsample(semantic, scale_factor=2, mode='nearest') 

        return semantic


def unit_test():
    
    mri = torch.tensor(np.random.rand(2,3,480,640).astype(np.float32))
    kn  = torch.tensor(np.random.rand(2,3,480,640).astype(np.float32))
    model = KGN(n_classes=2)
    y = model(mri, kn)
    print('output shape:', y.shape)
    assert y.shape == (2,2,480,640), 'output shape (2,2,480,640) is expected!'
    print('test ok!')
    torch.save(model, "KGN_parameters.pth")

if __name__ == '__main__':
    unit_test()