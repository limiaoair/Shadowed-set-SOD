from modulefinder import Module
from re import S
from turtle import forward
from matplotlib.backend_tools import ToolCursorPosition
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo

from Code.lib2.my2_module import Res2Net_Ours, DecomNet, ConvTranspose2D, Bottle2neck, BasicBlock, res2net101_v1b, ChannelAttention, CA_Block, CoordAtt
from Code.lib2.vgg import VGG16BN
from mmseg.models.backbones.swin import SwinTransformer
from Code.lib2.CSPNet import darknet53
import numpy as np
from Code.lib2.DRConv import MY_DRConv2d, MY_CoordAtt
from Code.lib2.model import *
import torchvision.transforms as transforms


def backbone_ResNet(pretrained, **kwargs):
    model_res = Res2Net_Ours(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    if pretrained:
        model_res.load_state_dict(torch.load('./Checkpoint/res/res2net50_v1b_26w_4s-3cf99910.pth'), strict=False)
        print("load backbone_resnet successfully!!!")
    return model_res 

def backbone_ResNet101(pretrained, **kwargs):
    model_res = Res2Net_Ours(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    if pretrained:
        model_res.load_state_dict(torch.load('./Checkpoint/res/res2net50_v1b_26w_4s-3cf99910.pth'), strict=False)
        print("load backbone_resnet successfully!!!")
    return model_res 

def backbone_decom(pretrained, **kwargs):
    model_de = DecomNet()
    if pretrained:
        model_de.load_state_dict(torch.load('./Checkpoint/Decom/9200.tar'), strict=False)
        print("load backbone_Decom successfully!!!")
    return model_de    

def backbone_swint(pretrained, **kwargs):
    """
    num_class=1000
    #     lib里面的swint:::    pretrained_model_path = 'H:/sod/SPNet/Checkpoint/swin/swin_base_patch4_window12_384_22kto1k.pth'
    #     mmseg::   H:/sod/SPNet/Checkpoint/swin/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth
    #     内置文件：  swin_tiny_patch4_window7_224.pth
    """
    # model_de = SwinTransformer(out_indices=(3,), frozen_stages=-1)
    # model_sw = SwinTransformer(in_chans=3,
    #                     patch_size=4,
    #                     window_size=7,
    #                     embed_dim=96,
    #                     depths=(2, 2, 6, 2),
    #                     num_heads=(3, 6, 12, 24),
    #                     num_class=1000,
    #                     **kwargs)
    model_sw = SwinTransformer(embed_dims=48,depths=(2, 2, 6),num_heads=(3, 6, 12, 24),strides=(4, 2, 2),out_indices=(0, 1, 2))
    if pretrained:
        model_sw.load_state_dict(torch.load('./Checkpoint/swin/swin_base_patch4_window12_384_22kto1k_mmseg.pth'), strict=False)
        print("load backbone_swint successfully!!!")
    return model_sw    

def backbone_vgg16(pretrained=True):
    model_vgg = VGG16BN()
    if pretrained:
        print("loading backbone_vgg successfully")
        model_vgg.load_state_dict(torch.load("./Checkpoint/vgg/vgg16.pth"), strict=False)
    return model_vgg

class I_ppp(nn.Module):
    def __init__(self, SCALE=224):
        super(I_ppp, self).__init__()
        self.pre1 = I_pre1(sca=0.8)
        self.pre2 = I_pre1(sca=0.6)
        self.pre3 = I_pre1(sca=0.4)
        self.con1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.con2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.ca = CoordAtt(inp=3, oup=3)
    
    def forward(self, x):
        y1 = self.pre1(x)
        y2 = self.pre2(x)
        y3 = self.pre3(x)
        out1 = torch.cat([y1, y2, y3], dim=1)
        # out1 = self.con1(out1)
        out = self.ca(out1)
        out = self.con2(out)
        return out

class I_pre1_ori(nn.Module):
    def __init__(self, sca):
        super(I_pre1_ori, self).__init__()
        self.scale = sca
        
    def forward(self, x1):
        new = torch.ones_like(x1)
        new = new*0.01
        out = torch.where(x1 > self.scale, new, x1)
        return out

class I_pre1(nn.Module):
    """
    shadowed set 其实就是标准化
    """
    def __init__(self, sca):
        super(I_pre1, self).__init__()
        self.scale = sca
        self.one = 1
        self.norm = transforms.Normalize(mean=[0], std=[1])
        
    def forward(self, x1):
        new = torch.ones_like(x1)
        new1 = torch.ones_like(x1)
        new = new*0.01
        new1 = new1*(self.scale)

        out = torch.where(x1 > self.scale, new1, x1)
        out = self.norm(out)
        a = torch.max(out)
        out = torch.where(out == a, new, out)
        out = torch.where(out < new, new, out)
        return out
  
    
class I_pre2(nn.Module):
    """
    shadowed set 其实就是归一化，这个是较为准确的代码
    """
    def __init__(self, sca):
        super(I_pre2, self).__init__()
        self.scale = sca
        self.one = 1
        self.norm = transforms.Normalize(mean=[0], std=[1])
        
    def forward(self, x1):
        new = torch.ones_like(x1)
        new1 = torch.ones_like(x1)
        new = new*0.01
        new1 = new1*(self.scale)

        out = torch.where(x1 > self.scale, new, x1)
        a = torch.max(out)
        b = torch.min(out)
        out = (out - a) / (b - a)
        out = torch.where(out < new, new, out)
        return out

class I_pre_process2(nn.Module):
    """
    最后的模块:MG-M+
    """
    def __init__(self):
        super(I_pre_process2, self).__init__()
        self.pre0 = I_pre2(sca=0.9)
        self.pre1 = I_pre2(sca=0.8)
        self.pre2 = I_pre2(sca=0.7)
        self.pre3 = I_pre2(sca=0.6)
        self.pre4 = I_pre2(sca=0.5)
        self.pre5 = I_pre2(sca=0.4)
        self.pre6 = I_pre2(sca=0.3)
        self.pre7 = I_pre2(sca=0.2)
        self.pre8 = I_pre2(sca=0.1)
        # self.diff = difference()
        self.con1 = nn.Conv2d(9, 3, kernel_size=3, stride=1, padding=1)
        self.con2 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y0 = self.pre0(x)
        y1 = self.pre1(x)
        y2 = self.pre2(x)
        y3 = self.pre3(x)
        y4 = self.pre4(x)
        y5 = self.pre5(x)
        y6 = self.pre6(x)
        y7 = self.pre7(x)
        y8 = self.pre8(x)
        # print(y2)
        y_list = [y0, y1, y2, y3, y4, y5, y6, y7, y8]
        a = [[0 for col in range(9)] for row in range(9)]
        for i in range(9):
            for j in range(9):
                a[i][j] = difference(y_list[i], y_list[j])
        r, c = np.where(a == np.min(a))
        out_1 = torch.cat([y_list[r[0]], y_list[c[0]]], dim=1)
        out_2 = y_list[0]
        for k in range(1, 9):
            out_2 = torch.cat([out_2, y_list[k]], dim=1)
        out_2 = self.con1(out_2)
        out_2 = self.con2(out_2)
        out = torch.cat([out_1, out_2], dim=1)
        return out


class I_pre_process(nn.Module):
    """
    最后的模块:MG-M+
    """
    def __init__(self):
        super(I_pre_process, self).__init__()
        self.pre0 = I_pre1(sca=0.9)
        self.pre1 = I_pre1(sca=0.8)
        self.pre2 = I_pre1(sca=0.7)
        self.pre3 = I_pre1(sca=0.6)
        self.pre4 = I_pre1(sca=0.5)
        self.pre5 = I_pre1(sca=0.4)
        self.pre6 = I_pre1(sca=0.3)
        self.pre7 = I_pre1(sca=0.2)
        self.pre8 = I_pre1(sca=0.1)
        # self.diff = difference()
        self.con1 = nn.Conv2d(9, 3, kernel_size=3, stride=1, padding=1)
        self.con2 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y0 = self.pre0(x)
        y1 = self.pre1(x)
        y2 = self.pre2(x)
        y3 = self.pre3(x)
        y4 = self.pre4(x)
        y5 = self.pre5(x)
        y6 = self.pre6(x)
        y7 = self.pre7(x)
        y8 = self.pre8(x)
        # print(y2)
        y_list = [y0, y1, y2, y3, y4, y5, y6, y7, y8]
        a = [[0 for col in range(9)] for row in range(9)]
        for i in range(9):
            for j in range(9):
                a[i][j] = difference(y_list[i], y_list[j])
        r, c = np.where(a == np.min(a))
        out_1 = torch.cat([y_list[r[0]], y_list[c[0]]], dim=1)
        out_2 = y_list[0]
        for k in range(1, 9):
            out_2 = torch.cat([out_2, y_list[k]], dim=1)
        out_2 = self.con1(out_2)
        out_2 = self.con2(out_2)
        out = torch.cat([out_1, out_2], dim=1)
        return out


def difference(in1, in2):
    ones1       = torch.ones_like(in1)
    zero1       = torch.zeros_like(in2)
    out1        = torch.where(in1 == in2, zero1, ones1)
    out_size    = torch.nonzero(out1)
    out2        = out_size.shape[0]
    out         = out2/zero1.numel()
    out         = math.sqrt((out-0.6615)**2)
    # out         = math.sqrt((out-0.6)**2)
    return out

class sod0502_dul_res(nn.Module):
    def __init__(self, ind=50):
        super(sod0502_dul_res, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = Dual_modal_Aggration_module(256, 128, 320)
        self.fu_2 = Dual_modal_Aggration_module(512, 256, 640)
        self.fu_3 = Dual_modal_Aggration_module(1024, 512, 1280)
        self.fu_4 = Dual_modal_Aggration_module(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1(1024, 256)
        self.c2             = C1(512, 128)
        self.c3             = C1(256, 64)
        self.c4             = C2(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())


    def forward(self, x, y):
        # print(y)
        # print(y.shape)                # [batchsize, 1, 352, 352]
        # print(torch.mean(y))
        # y0 = self.pp(y)
        y0 = self.pprecess(y)
        # print(y0)
        # print(asa)
        # y                    = torch.cat([y, y, y], dim=1)
        # print(y0.size()) 

        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        # print(i0.shape, i1.shape, i2.shape, i3.shape, i4.shape)
    
        ful_0    = self.fu_0(x0, i0)
        # print(ful_0.shape)
        ful_1    = self.fu_1(x1, i1, ful_0)
        # print(ful_1.shape)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        # print(ful_2.shape)
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        # [8, 64, 88, 88],[8, 128, 88, 88],[8, 256, 44, 44],[8, 512, 22, 22],[8, 1024, 11, 11])
        # print(ful_0.shape, ful_1.shape, ful_2.shape, ful_3.shape, ful_4.shape)
        C_0      = self.c0(ful_4)   
        # print(C_0.shape, ful_3.shape)  # [8, 512, 22, 22], [8, 512, 22, 22]
        C_1      = self.c1(C_0, ful_3)
        # print(C_1.shape, ful_2.shape)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape, ful_1.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape, ful_0.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)

        out      = self.ou(C_4)
        return out

class sod0502_dul_res_threeloss(nn.Module):
    """
    最后选用的模型 0605
    """
    def __init__(self, ind=50):
        super(sod0502_dul_res_threeloss, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = Dual_modal_Aggration_module(256, 128, 320)
        self.fu_2 = Dual_modal_Aggration_module(512, 256, 640)
        self.fu_3 = Dual_modal_Aggration_module(1024, 512, 1280)
        self.fu_4 = Dual_modal_Aggration_module(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1(1024, 256)
        self.c2             = C1(512, 128)
        self.c3             = C1(256, 64)
        self.c4             = C2(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):
        # x = torch.cat([y,y,y], 1)
        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1, ful_0)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0, ful_3)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0502_dul_res_three_pool(nn.Module):
    """
    弃用
    """
    def __init__(self, ind=50):
        super(sod0502_dul_res_three_pool, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0() 
        
        self.fu_1 = Dual_modal_Aggration_module(256, 128, 320)
        self.fu_2 = Dual_modal_Aggration_module(512, 256, 640)
        self.fu_3 = Dual_modal_Aggration_module(1024, 512, 1280)
        self.fu_4 = Dual_modal_Aggration_module(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1(1024, 256)
        self.c2             = C1(512, 128)
        self.c3             = C1(256, 64)
        self.c4             = C2(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):

        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1, ful_0)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0, ful_3)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0502_dul_ablation_DMA_CAT(nn.Module):
    def __init__(self, ind=50):
        super(sod0502_dul_ablation_DMA_CAT, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = DMA_ablation_cat(256, 128, 320)
        self.fu_2 = DMA_ablation_cat(512, 256, 640)
        self.fu_3 = DMA_ablation_cat(1024, 512, 1280)
        self.fu_4 = DMA_ablation_cat(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1(1024, 256)
        self.c2             = C1(512, 128)
        self.c3             = C1(256, 64)
        self.c4             = C2(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):

        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1, ful_0)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0, ful_3)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0502_dul_ablation_DMA_cross(nn.Module):
    def __init__(self, ind=50):
        super(sod0502_dul_ablation_DMA_cross, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = DMA_ablation_cross(256, 128, 320)
        self.fu_2 = DMA_ablation_cross(512, 256, 640)
        self.fu_3 = DMA_ablation_cross(1024, 512, 1280)
        self.fu_4 = DMA_ablation_cross(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1(1024, 256)
        self.c2             = C1(512, 128)
        self.c3             = C1(256, 64)
        self.c4             = C2(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):

        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1, ful_0)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0, ful_3)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0502_dul_ablation_DMA_skip(nn.Module):
    def __init__(self, ind=50):
        super(sod0502_dul_ablation_DMA_skip, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = DMA_ablation_skip(256, 128)
        self.fu_2 = DMA_ablation_skip(512, 256)
        self.fu_3 = DMA_ablation_skip(1024, 512)
        self.fu_4 = DMA_ablation_skip(2048, 1024)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1(1024, 256)
        self.c2             = C1(512, 128)
        self.c3             = C1(256, 64)
        self.c4             = C2(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):

        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1)
        ful_2    = self.fu_2(x2, i2)
        ful_3    = self.fu_3(x3, i3)
        ful_4    = self.fu_4(x4, i4)
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0, ful_3)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0502_dul_ablation_MGM(nn.Module):
    def __init__(self, ind=50):
        super(sod0502_dul_ablation_MGM, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_MGM_Ablation()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = Dual_modal_Aggration_module(256, 128, 320)
        self.fu_2 = Dual_modal_Aggration_module(512, 256, 640)
        self.fu_3 = Dual_modal_Aggration_module(1024, 512, 1280)
        self.fu_4 = Dual_modal_Aggration_module(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1(1024, 256)
        self.c2             = C1(512, 128)
        self.c3             = C1(256, 64)
        self.c4             = C2(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):

        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1, ful_0)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0, ful_3)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0502_dul_ablation_edge(nn.Module):
    """
    start-time:2023 06 10 23 02
    """
    def __init__(self, ind=50):
        super(sod0502_dul_ablation_edge, self).__init__()
        # self.pp             = I_ppp() 
        # self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = Dual_modal_Aggration_module(256, 128, 320)
        self.fu_2 = Dual_modal_Aggration_module(512, 256, 640)
        self.fu_3 = Dual_modal_Aggration_module(1024, 512, 1280)
        self.fu_4 = Dual_modal_Aggration_module(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1(1024, 256)
        self.c2             = C1(512, 128)
        self.c3             = C1(256, 64)
        self.c4             = C2(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):

        y0 = torch.cat((y, y, y), 1)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1, ful_0)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0, ful_3)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0502_dul_ablation_C(nn.Module):
    """
    c_no_cat
    start-time: 2023 06 09 19 51
    """
    def __init__(self, ind=50):
        super(sod0502_dul_ablation_C, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = Dual_modal_Aggration_module(256, 128, 320)
        self.fu_2 = Dual_modal_Aggration_module(512, 256, 640)
        self.fu_3 = Dual_modal_Aggration_module(1024, 512, 1280)
        self.fu_4 = Dual_modal_Aggration_module(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1_ablation(512, 256)
        self.c2             = C1_ablation(256, 128)
        self.c3             = C1_ablation(128, 64)
        self.c4             = C2_ablation(64, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):

        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1, ful_0)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0)
        C_2      = self.c2(C_1)
        # print(C_2.shape)
        C_3      = self.c3(C_2)
        # print(C_3.shape)
        C_4      = self.c4(C_3)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0502_dul_ablation_no_MGM(nn.Module):
    """
    start-time:2023 06 11 
    """
    def __init__(self, ind=50):
        super(sod0502_dul_ablation_no_MGM, self).__init__()
        # self.pp             = I_ppp() 
        # self.pprecess       = I_pre_MGM_Ablation()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = Dual_modal_Aggration_module(256, 128, 320)
        self.fu_2 = Dual_modal_Aggration_module(512, 256, 640)
        self.fu_3 = Dual_modal_Aggration_module(1024, 512, 1280)
        self.fu_4 = Dual_modal_Aggration_module(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1(1024, 256)
        self.c2             = C1(512, 128)
        self.c3             = C1(256, 64)
        self.c4             = C2(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):

        y0 = torch.cat((y, y, y), 1)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1, ful_0)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0, ful_3)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0502_dul_ablation_C_no_pool(nn.Module):
    """
    start-time:2023 06 12 21 05
    """
    def __init__(self, ind=50):
        super(sod0502_dul_ablation_C_no_pool, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back(3)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = Dual_modal_Aggration_module(256, 128, 320)
        self.fu_2 = Dual_modal_Aggration_module(512, 256, 640)
        self.fu_3 = Dual_modal_Aggration_module(1024, 512, 1280)
        self.fu_4 = Dual_modal_Aggration_module(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1_ablation_no_pool(1024, 256)
        self.c2             = C1_ablation_no_pool(512, 128)
        self.c3             = C1_ablation_no_pool(256, 64)
        self.c4             = C2_ablation_no_pool(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):

        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1, ful_0)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0, ful_3)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0502_dul_ablation_Same_backbone(nn.Module):
    """
    最后选用的模型 0605
    """
    def __init__(self, ind=50):
        super(sod0502_dul_ablation_Same_backbone, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = backbone_ResNet(ind)

        self.fu_0 = Dual_modal_Aggration_module0()#
        
        self.fu_1 = Dual_modal_Aggration_module(256, 128, 320)
        self.fu_2 = Dual_modal_Aggration_module(512, 256, 640)
        self.fu_3 = Dual_modal_Aggration_module(1024, 512, 1280)
        self.fu_4 = Dual_modal_Aggration_module(2048, 1024, 2560)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(1024, 512)
        self.c1             = C1(1024, 256)
        self.c2             = C1(512, 128)
        self.c3             = C1(256, 64)
        self.c4             = C2(128, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):

        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)    # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)
        ful_0    = self.fu_0(x0, i0)
        ful_1    = self.fu_1(x1, i1, ful_0)
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))
        C_0      = self.c0(ful_4)   
        C_1      = self.c1(C_0, ful_3)
        C_2      = self.c2(C_1, ful_2)
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class sod0814_end_light_ablation_thermal(nn.Module):
    """
    最后选用的模型 用原来的pre来进行light
    channel_light 适当增大channel
    """
    def __init__(self, ind=50):
        super(sod0814_end_light_ablation_thermal, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back_light_end(3, 32, 32, 64, 128, 128)

        self.chan_trans1    = tran_chan_end(32, 32, 64, 128, 128)
        # self.chan_trans2    = tran_chan()

        self.fu_0 = Dual_modal_Aggration_module0_light_end(32, 32)#

        self.fu_1 = Dual_modal_Aggration_module_light_end(32, 32, 32)
        self.fu_2 = Dual_modal_Aggration_module_light_end(64, 32, 32)
        self.fu_3 = Dual_modal_Aggration_module_light_end(128, 64, 32)
        self.fu_4 = Dual_modal_Aggration_module_light_end(128, 128, 64)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(128, 32)
        self.c1             = C1(96, 32)
        self.c2             = C1(64, 32)
        self.c3             = C1(64, 32)
        self.c4             = C2_light(32, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):
        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)
           
        # torch.Size([10, 64, 88, 88],[10, 256, 88, 88],[10, 512, 44, 44],[10, 1024, 22, 22],[10, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)

        x0, x1, x2, x3, x4     = self.chan_trans1(x0, x1, x2, x3, x4)
        # i0, i1, i2, i3, i4     = self.chan_trans1(i0, i1, i2, i3, i4)
        # print(x0.shape,x1.shape,x2.shape,x3.shape,x4.shape)
        # print(i0.shape,i1.shape,i2.shape,i3.shape,i4.shape)
        # torch.Size([14, 16, 88, 88],[14, 16, 88, 88],[14, 32, 44, 44],[14, 64, 22, 22],[14, 128, 11, 11])
        
        ful_0    = self.fu_0(x0, i0)  # torch.Size([14, 16, 88, 88])
        ful_1    = self.fu_1(x1, i1, ful_0)  # torch.Size([14, 16, 88, 88])
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))  # torch.Size([14, 16, 44, 44])
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))  # torch.Size([14, 32, 22, 22])
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))  # torch.Size([14, 32, 11, 11])
        C_0      = self.c0(ful_4)  # torch.Size([14, 16, 22, 22])
        C_1      = self.c1(C_0, ful_3)  # torch.Size([14, 16, 44, 44])
        C_2      = self.c2(C_1, ful_2)  # torch.Size([14, 16, 88, 88])
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)  # torch.Size([14, 16, 176, 176])
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)  # torch.Size([14, 16, 176, 176])
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3


class sod0814_end_light_ablation_same_backbone_nopre(nn.Module):
    def __init__(self, ind=50):
        super(sod0814_end_light_ablation_same_backbone_nopre, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back_light_end(3, 32, 32, 64, 128, 128)

        self.chan_trans1    = tran_chan_end(32, 32, 64, 128, 128)
        # self.chan_trans2    = tran_chan()

        self.fu_0 = Dual_modal_Aggration_module0_light_end(32, 32)#

        self.fu_1 = Dual_modal_Aggration_module_light_end(32, 32, 32)
        self.fu_2 = Dual_modal_Aggration_module_light_end(64, 32, 32)
        self.fu_3 = Dual_modal_Aggration_module_light_end(128, 64, 32)
        self.fu_4 = Dual_modal_Aggration_module_light_end(128, 128, 64)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(128, 32)
        self.c1             = C1(96, 32)
        self.c2             = C1(64, 32)
        self.c3             = C1(64, 32)
        self.c4             = C2_light(32, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):
        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)
           
        # torch.Size([10, 64, 88, 88],[10, 256, 88, 88],[10, 512, 44, 44],[10, 1024, 22, 22],[10, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)

        x0, x1, x2, x3, x4     = self.chan_trans1(x0, x1, x2, x3, x4)
        # i0, i1, i2, i3, i4     = self.chan_trans1(i0, i1, i2, i3, i4)
        # print(x0.shape,x1.shape,x2.shape,x3.shape,x4.shape)
        # print(i0.shape,i1.shape,i2.shape,i3.shape,i4.shape)
        # torch.Size([14, 16, 88, 88],[14, 16, 88, 88],[14, 32, 44, 44],[14, 64, 22, 22],[14, 128, 11, 11])
        
        ful_0    = self.fu_0(x0, i0)  # torch.Size([14, 16, 88, 88])
        ful_1    = self.fu_1(x1, i1, ful_0)  # torch.Size([14, 16, 88, 88])
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))  # torch.Size([14, 16, 44, 44])
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))  # torch.Size([14, 32, 22, 22])
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))  # torch.Size([14, 32, 11, 11])
        C_0      = self.c0(ful_4)  # torch.Size([14, 16, 22, 22])
        C_1      = self.c1(C_0, ful_3)  # torch.Size([14, 16, 44, 44])
        C_2      = self.c2(C_1, ful_2)  # torch.Size([14, 16, 88, 88])
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)  # torch.Size([14, 16, 176, 176])
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)  # torch.Size([14, 16, 176, 176])
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3



class C0(nn.Module):
    def __init__(self, x, y):
        super(C0, self).__init__()
        self.Conv0 = nn.Sequential(nn.Conv2d(x, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())      
        self.Conv1 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.Conv2 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up    = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())

    def forward(self, f):
        p0 = self.Conv0(f)
        p1 = self.Conv1(p0)
        p2 = self.Conv2(p1)
        p3 = self.up(p2)
        return p3 

class C1(nn.Module):
    def __init__(self, x, y):
        super(C1, self).__init__()
        self.Conv0 = nn.Sequential(nn.Conv2d(x, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())      
        self.Conv1 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv2 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up1   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv3 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up2   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv4 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 

    def forward(self, f, v):
        p00      = torch.cat((f, v), 1)
        # print('1', p00.shape)
        p0      = self.Conv0(p00)
        p1      = self.Conv1(p0)
        p_1     = self.Pool1(p1)
        # print(p_1.shape, "222")
        p2      = self.Conv2(p_1)
        p_up1   = self.up1(p2)
        # print(p_up1.shape, "222")
        p3      = self.Conv3(p_up1)
        # print(p3.shape, "3323")
        p_up2   = self.up2(p3)
        # print(p_up2.shape, "333")
        out     = self.Conv4(p_up2)
        return out 
    
class C2(nn.Module):
    def __init__(self, x, y):
        super(C2, self).__init__()
        self.Conv0 = nn.Sequential(nn.Conv2d(64, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())      
        self.Conv1 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv2 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.PReLU())
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv3 = nn.Sequential(nn.Conv2d(64, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.up1   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv4 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up2   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv5 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 

    def forward(self, f, v):
        p0      = self.Conv0(f)
        p1      = self.Conv1(p0)
        p_1     = self.Pool1(p1)
        p_cat   = torch.cat((p_1, v), 1)
        # print(p_cat.shape, "p_cat")
        p2      = self.Conv2(p_cat)
        # print(p2.shape, "p2")
        p_2     = self.Pool2(p2)
        # print(p_2.shape, "p_2")
        p3      = self.Conv3(p_2)
        # print(p3.shape, "3323")
        p_up1   = self.up1(p3)
        # print(p_up1.shape, "222")
        p4      = self.Conv4(p_up1)
        
        p_up2   = self.up2(p4)
        # print(p_up2.shape, "333")
        out     = self.Conv5(p_up2)
        # print(out.shape, "out")
        return out 
   
class i_back(nn.Module):
    def __init__(self, ind):
        super(i_back, self).__init__()
        # 352 - 176 - 88 - 44 - 22 - 11
        # 64 88 torch.Size([8, 256, 88, 88], [8, 512, 44, 44], [8, 1024, 22, 22], [8, 2048, 11, 11])
        self.conv1 = nn.Sequential(nn.Conv2d(ind, 32, 3, 1, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
                                   nn.Conv2d(32, 32, 3, 2, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.att1  = nn.Sequential(MY_CoordAtt(32, 32),
                                   nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.PReLU())
        self.att2  = nn.Sequential(MY_CoordAtt(64, 64),
                                   nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), nn.PReLU(),
                                #    nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(128), nn.PReLU(),
                                   nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.PReLU())
        
        self.conv3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.PReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(256), nn.PReLU())
        self.att3  = nn.Sequential(MY_CoordAtt(256, 256),
                                   nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.PReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.PReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.PReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(512), nn.PReLU())
        self.att4  = nn.Sequential(MY_CoordAtt(512, 512),
                                #    nn.Conv2d(512, 512, kernel_size=1, padding=2, stride=1, dilation=1), nn.BatchNorm2d(32), nn.PReLU(),
                                   nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1024), nn.PReLU(),
                                   nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1024), nn.PReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1024), nn.PReLU(),
                                   nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(1024), nn.PReLU())
        self.att5  = nn.Sequential(MY_CoordAtt(1024, 1024),
                                   nn.Conv2d(1024, 2048, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(2048), nn.PReLU(),
                                   nn.Conv2d(2048, 2048, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(2048), nn.PReLU())

    def forward(self, f):
        p1      = self.conv1(f)
        # print('11', f.shape, p1.shape)
        out1    = self.att1(p1)
        # print(out1.shape)

        p2      = self.conv2(out1)
        out2    = self.att2(p2)
        # print(out2.shape)

        p3      = self.conv3(out2)
        out3    = self.att3(p3)
        # print(out3.shape)

        p4      = self.conv4(out3)
        out4    = self.att4(p4)
        # print(out4.shape)

        p5      = self.conv5(out4)
        # print(p5.shape)
        out5    = self.att5(p5)
        # print(out5.shape)
        return out1, out2, out3, out4, out5



class C1_ablation(nn.Module):
    def __init__(self, x, y):
        super(C1_ablation, self).__init__()
        self.Conv0 = nn.Sequential(nn.Conv2d(x, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())      
        self.Conv1 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv2 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up1   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv3 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up2   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv4 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 

    def forward(self, f):
        # p00      = torch.cat((f, v), 1)
        # print('1', p00.shape)
        p0      = self.Conv0(f)
        p1      = self.Conv1(p0)
        p_1     = self.Pool1(p1)
        # print(p_1.shape, "222")
        p2      = self.Conv2(p_1)
        p_up1   = self.up1(p2)
        # print(p_up1.shape, "222")
        p3      = self.Conv3(p_up1)
        # print(p3.shape, "3323")
        p_up2   = self.up2(p3)
        # print(p_up2.shape, "333")
        out     = self.Conv4(p_up2)
        return out 
class C2_ablation(nn.Module):
    def __init__(self, x, y):
        super(C2_ablation, self).__init__()
        self.Conv0 = nn.Sequential(nn.Conv2d(64, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())      
        self.Conv1 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.PReLU())
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv3 = nn.Sequential(nn.Conv2d(64, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.up1   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv4 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up2   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv5 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 

    def forward(self, f):
        p0      = self.Conv0(f)
        p1      = self.Conv1(p0)
        p_1     = self.Pool1(p1)
        # p_cat   = torch.cat((p_1, v), 1)
        # print(p_cat.shape, "p_cat")
        p2      = self.Conv2(p_1)
        # print(p2.shape, "p2")
        p_2     = self.Pool2(p2)
        # print(p_2.shape, "p_2")
        p3      = self.Conv3(p_2)
        # print(p3.shape, "3323")
        p_up1   = self.up1(p3)
        # print(p_up1.shape, "222")
        p4      = self.Conv4(p_up1)
        
        p_up2   = self.up2(p4)
        # print(p_up2.shape, "333")
        out     = self.Conv5(p_up2)
        # print(out.shape, "out")
        return out 

class C1_ablation_no_pool(nn.Module):
    def __init__(self, x, y):
        super(C1_ablation_no_pool, self).__init__()
        self.Conv0 = nn.Sequential(nn.Conv2d(x, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())      
        self.Conv1 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv2 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up1   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv3 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up2   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv4 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 

    def forward(self, f, v):
        p00      = torch.cat((f, v), 1)
        # print('1', p00.shape)
        p0      = self.Conv0(p00)
        p1      = self.Conv1(p0)
        # p_1     = self.Pool1(p1)
        # print(p_1.shape, "222")
        p2      = self.Conv2(p1)
        # p_up1   = self.up1(p2)
        # print(p_up1.shape, "222")
        p3      = self.Conv3(p2)
        # print(p3.shape, "3323")
        p_up2   = self.up2(p3)
        # print(p_up2.shape, "333")
        out     = self.Conv4(p_up2)
        return out   
class C2_ablation_no_pool(nn.Module):
    def __init__(self, x, y):
        super(C2_ablation_no_pool, self).__init__()
        self.Conv0 = nn.Sequential(nn.Conv2d(64, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())      
        self.Conv1 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv2 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.PReLU())
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv3 = nn.Sequential(nn.Conv2d(64, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.up1   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv4 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up2   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv5 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 

    def forward(self, f, v):
        p0      = self.Conv0(f)
        p1      = self.Conv1(p0)
        p_1     = self.Pool1(p1)
        p_cat   = torch.cat((p_1, v), 1)
        # print(p_cat.shape, "p_cat")
        p2      = self.Conv2(p_cat)
        # print(p2.shape, "p2")
        # p_2     = self.Pool2(p2)
        # print(p_2.shape, "p_2")
        p3      = self.Conv3(p2)
        # print(p3.shape, "3323")
        # p_up1   = self.up1(p3)
        # print(p_up1.shape, "222")
        p4      = self.Conv4(p3)
        
        p_up2   = self.up2(p4)
        # print(p_up2.shape, "333")
        out     = self.Conv5(p_up2)
        # print(out.shape, "out")
        return out 
  
class C1_ablation_no_two(nn.Module):
    def __init__(self, x, y):
        super(C1_ablation_no_two, self).__init__()
        self.Conv0 = nn.Sequential(nn.Conv2d(x, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())      
        self.Conv1 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv2 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up1   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv3 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up2   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv4 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 

    def forward(self, f):
        # p00      = torch.cat((f, v), 1)
        # print('1', p00.shape)
        p0      = self.Conv0(f)
        p1      = self.Conv1(p0)
        # p_1     = self.Pool1(p1)
        # print(p_1.shape, "222")
        p2      = self.Conv2(p1)
        # p_up1   = self.up1(p2)
        # print(p_up1.shape, "222")
        p3      = self.Conv3(p2)
        # print(p3.shape, "3323")
        p_up2   = self.up2(p3)
        # print(p_up2.shape, "333")
        out     = self.Conv4(p_up2)
        return out   
class C2_ablation_no_two(nn.Module):
    def __init__(self, x, y):
        super(C2_ablation_no_two, self).__init__()
        self.Conv0 = nn.Sequential(nn.Conv2d(32, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())      
        self.Conv1 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.PReLU())
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv3 = nn.Sequential(nn.Conv2d(64, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.up1   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv4 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up2   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv5 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 

    def forward(self, f):
        p0      = self.Conv0(f)
        p1      = self.Conv1(p0)
        p_1     = self.Pool1(p1)
        # p_cat   = torch.cat((p_1), 1)
        # print(p_cat.shape, "p_cat")
        p2      = self.Conv2(p_1)
        # print(p2.shape, "p2")
        # p_2     = self.Pool2(p2)
        # print(p_2.shape, "p_2")
        p3      = self.Conv3(p2)
        # print(p3.shape, "3323")
        # p_up1   = self.up1(p3)
        # print(p_up1.shape, "222")
        p4      = self.Conv4(p3)
        
        p_up2   = self.up2(p4)
        # print(p_up2.shape, "333")
        out     = self.Conv5(p_up2)
        # print(out.shape, "out")
        return out 
  

class I_pre_MGM_Ablation(nn.Module):
    def __init__(self):
        super(I_pre_MGM_Ablation, self).__init__()
        self.pre0 = I_pre1(sca=0.9)
        self.pre1 = I_pre1(sca=0.8)
        self.pre2 = I_pre1(sca=0.7)
        self.pre3 = I_pre1(sca=0.6)
        self.pre4 = I_pre1(sca=0.5)
        self.pre5 = I_pre1(sca=0.4)
        self.pre6 = I_pre1(sca=0.3)
        self.pre7 = I_pre1(sca=0.2)
        self.pre8 = I_pre1(sca=0.1)
        # self.diff = difference()
        self.con1 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
        self.con2 = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y0 = self.pre0(x)
        y1 = self.pre1(x)
        y2 = self.pre2(x)
        y3 = self.pre3(x)
        y4 = self.pre4(x)
        y5 = self.pre5(x)
        y6 = self.pre6(x)
        y7 = self.pre7(x)
        y8 = self.pre8(x)
        y_list = [y0, y1, y2, y3, y4, y5, y6, y7, y8]
        a = [[0 for col in range(9)] for row in range(9)]
        for i in range(9):
            for j in range(9):
                a[i][j] = difference(y_list[i], y_list[j])
        r, c = np.where(a == np.min(a))
        out_1 = torch.cat([y_list[r[0]], y_list[c[0]]], dim=1)
        # out_2 = y_list[0]
        # for k in range(1, 9):
        #     out_2 = torch.cat([out_2, y_list[k]], dim=1)
        out_2 = self.con1(out_1)
        out = self.con2(out_2)
        # out = torch.cat([out_1, out_2], dim=1)
        return out

class Dual_modal_Aggration_module0(nn.Module):    
    def __init__(self, out_dim=64): 
        super(Dual_modal_Aggration_module0, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        

        self.layer_10 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.layer_20 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))

        self.layer_101 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_202 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_301 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.layer_302 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

        self.layer_303 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_304 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
         
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_22 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

        self.layer_33 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)       

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*3, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth):
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)

        x_rgb_s = self.layer_101(rgb)
        x_dep_s = self.layer_202(depth)

        rgb_w = nn.Sigmoid()(x_rgb_s)
        dep_w = nn.Sigmoid()(x_dep_s)

        Rgb_f = x_rgb.mul(dep_w)
        i_f   = x_dep.mul(rgb_w)

        Rgb_f_c1 = self.layer_301(Rgb_f)
        Rgb_f_c2 = self.layer_302(Rgb_f) 

        i_f_c1 = self.layer_303(i_f)
        i_f_c2 = self.layer_304(i_f)

        up_cat   = torch.cat((Rgb_f_c1, i_f_c1),dim=1)
        do_cat   = torch.cat((Rgb_f_c2, i_f_c2),dim=1)

        x_ca_up = self.layer_11(up_cat)
        x_ca_do = self.layer_22(do_cat)

        ou_cat   = torch.cat((x_ca_up, x_ca_do), dim=1)
        out    = self.layer_33(ou_cat)
        return out

class Dual_modal_Aggration_module(nn.Module):    
    def __init__(self, in_dim, out_dim, m_dim): 
        super(Dual_modal_Aggration_module, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        
        self.layer_10 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.layer_20 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))

        self.layer_101 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_202 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_301 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.layer_302 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

        self.layer_303 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_304 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
         
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_22 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

        self.layer_33 = nn.Sequential(nn.Conv2d(m_dim, out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim*2),act_fn)       

        self.layer_44 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth, xx):
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)

        x_rgb_s = self.layer_101(rgb)
        x_dep_s = self.layer_202(depth)

        rgb_w = nn.Sigmoid()(x_rgb_s)
        dep_w = nn.Sigmoid()(x_dep_s)

        Rgb_f = x_rgb.mul(dep_w)
        i_f   = x_dep.mul(rgb_w)

        Rgb_f_c1 = self.layer_301(Rgb_f)
        Rgb_f_c2 = self.layer_302(Rgb_f) 

        i_f_c1 = self.layer_303(i_f)
        i_f_c2 = self.layer_304(i_f)

        up_cat   = torch.cat((Rgb_f_c1, i_f_c1),dim=1)
        do_cat   = torch.cat((Rgb_f_c2, i_f_c2),dim=1)

        x_ca_up = self.layer_11(up_cat)
        x_ca_do = self.layer_22(do_cat)

        ou_cat   = torch.cat((x_ca_up, x_ca_do, xx), dim=1)
        out    = self.layer_44(self.layer_33(ou_cat))
        # out    = self.layer_44(self.layer_33(out))
        return out

class DMA_ablation_cat(nn.Module):    
    def __init__(self, in_dim, out_dim, m_dim): 
        super(DMA_ablation_cat, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        
        self.layer_10 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.layer_20 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))

        self.layer_101 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_202 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_301 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.layer_302 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

        self.layer_303 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_304 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
         
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_22 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

        self.layer_33 = nn.Sequential(nn.Conv2d(m_dim, out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim*2),act_fn)       

        self.layer_44 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth, xx):
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)

        x_rgb_s = self.layer_101(rgb)
        x_dep_s = self.layer_202(depth)

        rgb_w = nn.Sigmoid()(x_rgb_s)
        dep_w = nn.Sigmoid()(x_dep_s)

        Rgb_f = x_rgb.mul(dep_w)
        i_f   = x_dep.mul(rgb_w)

        Rgb_f_c1 = self.layer_301(Rgb_f)
        Rgb_f_c2 = self.layer_302(Rgb_f) 

        # i_f_c1 = self.layer_303(i_f)
        # i_f_c2 = self.layer_304(i_f)

        # up_cat   = torch.cat((Rgb_f_c1, i_f_c1),dim=1)
        # do_cat   = torch.cat((Rgb_f_c2, i_f_c2),dim=1)

        # x_ca_up = self.layer_11(up_cat)
        # x_ca_do = self.layer_22(do_cat)

        ou_cat   = torch.cat((Rgb_f_c1, Rgb_f_c2, xx), dim=1)
        out    = self.layer_44(self.layer_33(ou_cat))
        # out    = self.layer_44(self.layer_33(out))
        return out

class DMA_ablation_cross(nn.Module):    
    def __init__(self, in_dim, out_dim, m_dim): 
        super(DMA_ablation_cross, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        
        self.layer_10 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.layer_20 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))

        # self.layer_101 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        # self.layer_202 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_301 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.layer_302 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

        self.layer_303 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_304 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
         
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_22 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

        self.layer_33 = nn.Sequential(nn.Conv2d(m_dim, out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim*2),act_fn)       

        self.layer_44 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth, xx):
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)

        # x_rgb_s = self.layer_101(rgb)
        # x_dep_s = self.layer_202(depth)

        # rgb_w = nn.Sigmoid()(x_rgb_s)
        # dep_w = nn.Sigmoid()(x_dep_s)

        # Rgb_f = x_rgb.mul(dep_w)
        # i_f   = x_dep.mul(rgb_w)

        Rgb_f_c1 = self.layer_301(x_rgb)
        Rgb_f_c2 = self.layer_302(x_rgb) 

        i_f_c1 = self.layer_303(x_dep)
        i_f_c2 = self.layer_304(x_dep)

        up_cat   = torch.cat((Rgb_f_c1, i_f_c1),dim=1)
        do_cat   = torch.cat((Rgb_f_c2, i_f_c2),dim=1)

        x_ca_up = self.layer_11(up_cat)
        x_ca_do = self.layer_22(do_cat)

        ou_cat   = torch.cat((x_ca_up, x_ca_do, xx), dim=1)
        out    = self.layer_44(self.layer_33(ou_cat))
        # out    = self.layer_44(self.layer_33(out))
        return out

class DMA_ablation_skip(nn.Module):    
    def __init__(self, in_dim, out_dim): 
        super(DMA_ablation_skip, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        
        self.layer_10 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.layer_20 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))

        self.layer_101 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_202 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_301 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.layer_302 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

        self.layer_303 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_304 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
         
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_22 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

        self.layer_33 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)       

        self.layer_44 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth):
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)

        x_rgb_s = self.layer_101(rgb)
        x_dep_s = self.layer_202(depth)

        rgb_w = nn.Sigmoid()(x_rgb_s)
        dep_w = nn.Sigmoid()(x_dep_s)

        Rgb_f = x_rgb.mul(dep_w)
        i_f   = x_dep.mul(rgb_w)

        Rgb_f_c1 = self.layer_301(Rgb_f)
        Rgb_f_c2 = self.layer_302(Rgb_f) 

        i_f_c1 = self.layer_303(i_f)
        i_f_c2 = self.layer_304(i_f)

        up_cat   = torch.cat((Rgb_f_c1, i_f_c1),dim=1)
        do_cat   = torch.cat((Rgb_f_c2, i_f_c2),dim=1)

        x_ca_up = self.layer_11(up_cat)
        x_ca_do = self.layer_22(do_cat)

        ou_cat   = torch.cat((x_ca_up, x_ca_do), dim=1)
        out    = self.layer_44(self.layer_33(ou_cat))
        # out    = self.layer_44(self.layer_33(out))
        return out

# ******************************************************************

class sod0502_end_light(nn.Module):
    """
    最后选用的模型 0605_light
    channel_light
    """
    def __init__(self, ind=50):
        super(sod0502_end_light, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process2()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back_light(3)

        self.chan_trans1    = tran_chan()
        # self.chan_trans2    = tran_chan()

        self.fu_0 = Dual_modal_Aggration_module0_light()#

        self.fu_1 = Dual_modal_Aggration_module_light(16, 16, 48)
        self.fu_2 = Dual_modal_Aggration_module_light(32, 16, 48)
        self.fu_3 = Dual_modal_Aggration_module_light(64, 32, 80)
        self.fu_4 = Dual_modal_Aggration_module_light(128, 32, 96)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(32, 16)
        self.c1             = C1(48, 16)
        self.c2             = C1(32, 16)
        self.c3             = C1(32, 16)
        self.c4             = C2_light(16, 16)

        self.ou             = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):
        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)
           
        # torch.Size([10, 64, 88, 88],[10, 256, 88, 88],[10, 512, 44, 44],[10, 1024, 22, 22],[10, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)

        x0, x1, x2, x3, x4     = self.chan_trans1(x0, x1, x2, x3, x4)
        # i0, i1, i2, i3, i4     = self.chan_trans1(i0, i1, i2, i3, i4)
        # print(x0.shape,x1.shape,x2.shape,x3.shape,x4.shape)
        # print(i0.shape,i1.shape,i2.shape,i3.shape,i4.shape)
        # torch.Size([14, 16, 88, 88],[14, 16, 88, 88],[14, 32, 44, 44],[14, 64, 22, 22],[14, 128, 11, 11])
        
        ful_0    = self.fu_0(x0, i0)  # torch.Size([14, 16, 88, 88])
        ful_1    = self.fu_1(x1, i1, ful_0)  # torch.Size([14, 16, 88, 88])
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))  # torch.Size([14, 16, 44, 44])
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))  # torch.Size([14, 32, 22, 22])
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))  # torch.Size([14, 32, 11, 11])
        C_0      = self.c0(ful_4)  # torch.Size([14, 16, 22, 22])
        C_1      = self.c1(C_0, ful_3)  # torch.Size([14, 16, 44, 44])
        C_2      = self.c2(C_1, ful_2)  # torch.Size([14, 16, 88, 88])
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)  # torch.Size([14, 16, 176, 176])
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)  # torch.Size([14, 16, 176, 176])
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3

class tran_chan(nn.Module):
    def __init__(self):
        super(tran_chan, self).__init__()
        self.trans_channel4  = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), nn.PReLU())
        self.trans_channel3  = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.PReLU())
        self.trans_channel2  = nn.Sequential(nn.Conv2d(512, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU())
        self.trans_channel1  = nn.Sequential(nn.Conv2d(256, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU())
        self.trans_channel0  = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU())

    def forward(self, x0, x1, x2, x3, x4):
        x0  =  self.trans_channel0(x0)
        x1  =  self.trans_channel1(x1)
        x2  =  self.trans_channel2(x2)
        x3  =  self.trans_channel3(x3)
        x4  =  self.trans_channel4(x4)

        return x0, x1, x2, x3, x4

class Dual_modal_Aggration_module0_light(nn.Module):    
    def __init__(self, out_dim=16): 
        super(Dual_modal_Aggration_module0_light, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        

        self.layer_10 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.layer_20 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))

        self.layer_101 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_202 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_301 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.layer_302 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

        self.layer_303 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_304 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
         
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_22 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

        self.layer_33 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)       

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*3, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth):
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)

        x_rgb_s = self.layer_101(rgb)
        x_dep_s = self.layer_202(depth)

        rgb_w = nn.Sigmoid()(x_rgb_s)
        dep_w = nn.Sigmoid()(x_dep_s)

        Rgb_f = x_rgb.mul(dep_w)
        i_f   = x_dep.mul(rgb_w)

        Rgb_f_c1 = self.layer_301(Rgb_f)
        Rgb_f_c2 = self.layer_302(Rgb_f) 

        i_f_c1 = self.layer_303(i_f)
        i_f_c2 = self.layer_304(i_f)

        up_cat   = torch.cat((Rgb_f_c1, i_f_c1),dim=1)
        do_cat   = torch.cat((Rgb_f_c2, i_f_c2),dim=1)

        x_ca_up = self.layer_11(up_cat)
        x_ca_do = self.layer_22(do_cat)

        ou_cat   = torch.cat((x_ca_up, x_ca_do), dim=1)
        out    = self.layer_33(ou_cat)
        return out

class Dual_modal_Aggration_module_light(nn.Module):    
    def __init__(self, in_dim, out_dim, m_dim): 
        super(Dual_modal_Aggration_module_light, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        
        self.layer_10 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.layer_20 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))

        self.layer_101 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_202 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_301 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.layer_302 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

        self.layer_303 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_304 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
         
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_22 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

        self.layer_33 = nn.Sequential(nn.Conv2d(m_dim, out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim*2),act_fn)       

        self.layer_44 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth, xx):
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)

        x_rgb_s = self.layer_101(rgb)
        x_dep_s = self.layer_202(depth)

        rgb_w = nn.Sigmoid()(x_rgb_s)
        dep_w = nn.Sigmoid()(x_dep_s)

        Rgb_f = x_rgb.mul(dep_w)
        i_f   = x_dep.mul(rgb_w)

        Rgb_f_c1 = self.layer_301(Rgb_f)
        Rgb_f_c2 = self.layer_302(Rgb_f) 

        i_f_c1 = self.layer_303(i_f)
        i_f_c2 = self.layer_304(i_f)

        up_cat   = torch.cat((Rgb_f_c1, i_f_c1),dim=1)
        do_cat   = torch.cat((Rgb_f_c2, i_f_c2),dim=1)

        x_ca_up = self.layer_11(up_cat)
        x_ca_do = self.layer_22(do_cat)

        ou_cat   = torch.cat((x_ca_up, x_ca_do, xx), dim=1)
        out    = self.layer_44(self.layer_33(ou_cat))
        # out    = self.layer_44(self.layer_33(out))
        return out


class C2_light(nn.Module):
    def __init__(self, x, y):
        super(C2_light, self).__init__()
        self.Conv0 = nn.Sequential(nn.Conv2d(x, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())      
        self.Conv1 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv2 = nn.Sequential(nn.Conv2d(x+y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv3 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU())
        self.up1   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv4 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 
        self.up2   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(y), nn.PReLU())
        self.Conv5 = nn.Sequential(nn.Conv2d(y, y, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(y), nn.PReLU()) 

    def forward(self, f, v):
        p0      = self.Conv0(f)
        p1      = self.Conv1(p0)
        p_1     = self.Pool1(p1)
        p_cat   = torch.cat((p_1, v), 1)  # torch.Size([14, 32, 88, 88]) p_cat
        # print(p_cat.shape, "p_cat")
        p2      = self.Conv2(p_cat)
        # print(p2.shape, "p2")
        p_2     = self.Pool2(p2)
        # print(p_2.shape, "p_2")
        p3      = self.Conv3(p_2)
        # print(p3.shape, "3323")
        p_up1   = self.up1(p3)
        # print(p_up1.shape, "222")
        p4      = self.Conv4(p_up1)
        
        p_up2   = self.up2(p4)
        # print(p_up2.shape, "333")
        out     = self.Conv5(p_up2)
        # print(out.shape, "out")
        return out 
   

class i_back_light(nn.Module):
    def __init__(self, ind):
        super(i_back_light, self).__init__()
        # 352 - 176 - 88 - 44 - 22 - 11
        # ([14, 16, 88, 88],[14, 16, 88, 88],[14, 32, 44, 44],[14, 64, 22, 22],[14, 128, 11, 11])
        self.conv1 = nn.Sequential(nn.Conv2d(ind, 16, 3, 1, padding=1), nn.BatchNorm2d(16), nn.PReLU(),
                                   nn.Conv2d(16, 16, 3, 2, padding=1), nn.BatchNorm2d(16), nn.PReLU())
        self.att1  = nn.Sequential(MY_CoordAtt(16, 16),
                                   nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.PReLU(),
                                   nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.PReLU())
        
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU())
        self.att2  = nn.Sequential(MY_CoordAtt(16, 16),
                                   nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                #    nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(128), nn.PReLU(),
                                   nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU())
        
        self.conv3 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                   nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(32), nn.PReLU())
        self.att3  = nn.Sequential(MY_CoordAtt(32, 32),
                                   nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                   nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                   nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(32), nn.PReLU())
        self.att4  = nn.Sequential(MY_CoordAtt(32, 32),
                                #    nn.Conv2d(512, 512, kernel_size=1, padding=2, stride=1, dilation=1), nn.BatchNorm2d(32), nn.PReLU(),
                                   nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                   nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.PReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.PReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(64), nn.PReLU())
        self.att5  = nn.Sequential(MY_CoordAtt(64, 64),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.PReLU(),
                                   nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), nn.PReLU())

    def forward(self, f):
        p1      = self.conv1(f)
        # print('11', f.shape, p1.shape)
        out1    = self.att1(p1)
        # print(out1.shape)

        p2      = self.conv2(out1)
        out2    = self.att2(p2)
        # print(out2.shape)

        p3      = self.conv3(out2)
        out3    = self.att3(p3)
        # print(out3.shape)

        p4      = self.conv4(out3)
        out4    = self.att4(p4)
        # print(out4.shape)

        p5      = self.conv5(out4)
        # print(p5.shape)
        out5    = self.att5(p5)
        # print(out5.shape)
        return out1, out2, out3, out4, out5


class sod0814_end_light(nn.Module):
    """
    最后选用的模型 用原来的pre来进行light
    channel_light 适当增大channel
    """
    def __init__(self, ind=50):
        super(sod0814_end_light, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back_light_end(3, 32, 32, 64, 128, 128)

        self.chan_trans1    = tran_chan_end(32, 32, 64, 128, 128)
        # self.chan_trans2    = tran_chan()

        self.fu_0 = Dual_modal_Aggration_module0_light_end(32, 32)#

        self.fu_1 = Dual_modal_Aggration_module_light_end(32, 32, 32)
        self.fu_2 = Dual_modal_Aggration_module_light_end(64, 32, 32)
        self.fu_3 = Dual_modal_Aggration_module_light_end(128, 64, 32)
        self.fu_4 = Dual_modal_Aggration_module_light_end(128, 128, 64)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(128, 32)
        self.c1             = C1(96, 32)
        self.c2             = C1(64, 32)
        self.c3             = C1(64, 32)
        self.c4             = C2_light(32, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):
        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)
           
        # torch.Size([10, 64, 88, 88],[10, 256, 88, 88],[10, 512, 44, 44],[10, 1024, 22, 22],[10, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)

        x0, x1, x2, x3, x4     = self.chan_trans1(x0, x1, x2, x3, x4)
        # i0, i1, i2, i3, i4     = self.chan_trans1(i0, i1, i2, i3, i4)
        # print(x0.shape,x1.shape,x2.shape,x3.shape,x4.shape)
        # print(i0.shape,i1.shape,i2.shape,i3.shape,i4.shape)
        # torch.Size([14, 16, 88, 88],[14, 16, 88, 88],[14, 32, 44, 44],[14, 64, 22, 22],[14, 128, 11, 11])
        
        ful_0    = self.fu_0(x0, i0)  # torch.Size([14, 16, 88, 88])
        ful_1    = self.fu_1(x1, i1, ful_0)  # torch.Size([14, 16, 88, 88])
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))  # torch.Size([14, 16, 44, 44])
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))  # torch.Size([14, 32, 22, 22])
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))  # torch.Size([14, 32, 11, 11])
        C_0      = self.c0(ful_4)  # torch.Size([14, 16, 22, 22])
        C_1      = self.c1(C_0, ful_3)  # torch.Size([14, 16, 44, 44])
        C_2      = self.c2(C_1, ful_2)  # torch.Size([14, 16, 88, 88])
        # print(C_2.shape)
        C_3      = self.c3(C_2, ful_1)  # torch.Size([14, 16, 176, 176])
        # print(C_3.shape)
        C_4      = self.c4(C_3, ful_0)  # torch.Size([14, 16, 176, 176])
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3


class sod0814_end_light_ablation_no_twoCCM(nn.Module):
    """
    最后选用的模型 用原来的pre来进行light
    channel_light 适当增大channel
    """
    def __init__(self, ind=50):
        super(sod0814_end_light_ablation_no_twoCCM, self).__init__()
        # self.pp             = I_ppp() 
        self.pprecess       = I_pre_process()
        self.layer_rgb      = backbone_ResNet(ind)
        self.layer_i        = i_back_light_end(3, 32, 32, 64, 128, 128)

        self.chan_trans1    = tran_chan_end(32, 32, 64, 128, 128)
        # self.chan_trans2    = tran_chan()

        self.fu_0 = Dual_modal_Aggration_module0_light_end(32, 32)#

        self.fu_1 = Dual_modal_Aggration_module_light_end(32, 32, 32)
        self.fu_2 = Dual_modal_Aggration_module_light_end(64, 32, 32)
        self.fu_3 = Dual_modal_Aggration_module_light_end(128, 64, 32)
        self.fu_4 = Dual_modal_Aggration_module_light_end(128, 128, 64)

        self.pool_fu_1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_fu_3      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.c0             = C0(128, 32)
        self.c1             = C1_ablation_no_two(32, 32)
        self.c2             = C1_ablation_no_two(32, 32)
        self.c3             = C1_ablation_no_two(32, 32)
        self.c4             = C2_ablation_no_two(32, 32)

        self.ou             = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou2            = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(3), nn.PReLU(),
                                            nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())
        self.ou3            = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(32), nn.PReLU(),
                                            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(16), nn.PReLU(),
                                            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.BatchNorm2d(8), nn.PReLU(),
                                            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(1), nn.PReLU())

    def forward(self, x, y):
        y0 = self.pprecess(y)
        x0, x1, x2, x3, x4    = self.layer_rgb(x)
           
        # torch.Size([10, 64, 88, 88],[10, 256, 88, 88],[10, 512, 44, 44],[10, 1024, 22, 22],[10, 2048, 11, 11])
        i0, i1, i2, i3, i4     = self.layer_i(y0)

        x0, x1, x2, x3, x4     = self.chan_trans1(x0, x1, x2, x3, x4)
        # i0, i1, i2, i3, i4     = self.chan_trans1(i0, i1, i2, i3, i4)
        # print(x0.shape,x1.shape,x2.shape,x3.shape,x4.shape)
        # print(i0.shape,i1.shape,i2.shape,i3.shape,i4.shape)
        # torch.Size([14, 16, 88, 88],[14, 16, 88, 88],[14, 32, 44, 44],[14, 64, 22, 22],[14, 128, 11, 11])
        
        ful_0    = self.fu_0(x0, i0)  # torch.Size([14, 16, 88, 88])
        ful_1    = self.fu_1(x1, i1, ful_0)  # torch.Size([14, 16, 88, 88])
        ful_2    = self.fu_2(x2, i2, self.pool_fu_1(ful_1))  # torch.Size([14, 16, 44, 44])
        ful_3    = self.fu_3(x3, i3, self.pool_fu_2(ful_2))  # torch.Size([14, 32, 22, 22])
        ful_4    = self.fu_4(x4, i4, self.pool_fu_3(ful_3))  # torch.Size([14, 32, 11, 11])
        C_0      = self.c0(ful_4)  # torch.Size([14, 16, 22, 22])
        C_1      = self.c1(C_0)  # torch.Size([14, 16, 44, 44])
        C_2      = self.c2(C_1)  # torch.Size([14, 16, 88, 88])
        # print(C_2.shape)
        C_3      = self.c3(C_2)  # torch.Size([14, 16, 176, 176])
        # print(C_3.shape)
        C_4      = self.c4(C_3)  # torch.Size([14, 16, 176, 176])
        # print(C_4.shape)
        out3     = self.ou3(C_2)
        out2     = self.ou2(C_3)
        out1     = self.ou(C_4)
        return out1, out2, out3



class tran_chan_end(nn.Module):
    def __init__(self, in1, in2, in3, in4, in5):
        super(tran_chan_end, self).__init__()
        self.trans_channel4  = nn.Sequential(nn.Conv2d(2048, in5, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in5), nn.PReLU())
        self.trans_channel3  = nn.Sequential(nn.Conv2d(1024, in4, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in4), nn.PReLU())
        self.trans_channel2  = nn.Sequential(nn.Conv2d(512, in3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in3), nn.PReLU())
        self.trans_channel1  = nn.Sequential(nn.Conv2d(256, in2, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in2), nn.PReLU())
        self.trans_channel0  = nn.Sequential(nn.Conv2d(64, in1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in1), nn.PReLU())

    def forward(self, x0, x1, x2, x3, x4):
        x0  =  self.trans_channel0(x0)
        x1  =  self.trans_channel1(x1)
        x2  =  self.trans_channel2(x2)
        x3  =  self.trans_channel3(x3)
        x4  =  self.trans_channel4(x4)

        return x0, x1, x2, x3, x4

class tran_chan_end_vgg(nn.Module):
    def __init__(self, in1, in2, in3, in4, in5):
        super(tran_chan_end_vgg, self).__init__()
        self.trans_channel4  = nn.Sequential(nn.Conv2d(512, in5, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in5), nn.PReLU())
        self.trans_channel3  = nn.Sequential(nn.Conv2d(512, in4, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in4), nn.PReLU())
        self.trans_channel2  = nn.Sequential(nn.Conv2d(256, in3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in3), nn.PReLU())
        self.trans_channel1  = nn.Sequential(nn.Conv2d(128, in2, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in2), nn.PReLU())
        self.trans_channel0  = nn.Sequential(nn.Conv2d(64, in1, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in1), nn.PReLU())

    def forward(self, x0, x1, x2, x3, x4):
        x0  =  self.trans_channel0(x0)
        x1  =  self.trans_channel1(x1)
        x2  =  self.trans_channel2(x2)
        x3  =  self.trans_channel3(x3)
        x4  =  self.trans_channel4(x4)

        return x0, x1, x2, x3, x4
    
class i_back_light_end(nn.Module):
    def __init__(self, ind, in1, in2, in3, in4, in5):
        super(i_back_light_end, self).__init__()
        # 352 - 176 - 88 - 44 - 22 - 11
        # ([14, 16, 88, 88],[14, 16, 88, 88],[14, 32, 44, 44],[14, 64, 22, 22],[14, 128, 11, 11])
        self.conv1 = nn.Sequential(nn.Conv2d(ind, in1, 3, 1, padding=1), nn.BatchNorm2d(in1), nn.PReLU(),
                                   nn.Conv2d(in1, in1, 3, 2, padding=1), nn.BatchNorm2d(in1), nn.PReLU())
        self.att1  = nn.Sequential(MY_CoordAtt(in1, in1),
                                   nn.Conv2d(in1, in1, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(in1), nn.PReLU(),
                                   nn.Conv2d(in1, in1, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in1), nn.PReLU())
        
        self.conv2 = nn.Sequential(nn.Conv2d(in1, in2, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in2), nn.PReLU())
        self.att2  = nn.Sequential(MY_CoordAtt(in2, in2),
                                   nn.Conv2d(in2, in2, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in2), nn.PReLU(),
                                #    nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(128), nn.PReLU(),
                                   nn.Conv2d(in2, in2, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in2), nn.PReLU())
        
        self.conv3 = nn.Sequential(nn.Conv2d(in2, in2, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in2), nn.PReLU(),
                                   nn.Conv2d(in2, in3, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(in3), nn.PReLU())
        self.att3  = nn.Sequential(MY_CoordAtt(in3, in3),
                                   nn.Conv2d(in3, in3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in3), nn.PReLU(),
                                   nn.Conv2d(in3, in3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in3), nn.PReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(in3, in3, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in3), nn.PReLU(),
                                   nn.Conv2d(in3, in4, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(in4), nn.PReLU())
        self.att4  = nn.Sequential(MY_CoordAtt(in4, in4),
                                #    nn.Conv2d(512, 512, kernel_size=1, padding=2, stride=1, dilation=1), nn.BatchNorm2d(32), nn.PReLU(),
                                   nn.Conv2d(in4, in4, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in4), nn.PReLU(),
                                   nn.Conv2d(in4, in4, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in4), nn.PReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(in4, in4, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in4), nn.PReLU(),
                                   nn.Conv2d(in4, in5, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(in5), nn.PReLU())
        self.att5  = nn.Sequential(MY_CoordAtt(in5, in5),
                                   nn.Conv2d(in5, in5, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in5), nn.PReLU(),
                                   nn.Conv2d(in5, in5, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in5), nn.PReLU())

    def forward(self, f):
        p1      = self.conv1(f)
        # print('11', f.shape, p1.shape)
        out1    = self.att1(p1)
        # print(out1.shape)

        p2      = self.conv2(out1)
        out2    = self.att2(p2)
        # print(out2.shape)

        p3      = self.conv3(out2)
        out3    = self.att3(p3)
        # print(out3.shape)

        p4      = self.conv4(out3)
        out4    = self.att4(p4)
        # print(out4.shape)

        p5      = self.conv5(out4)
        # print(p5.shape)
        out5    = self.att5(p5)
        # print(out5.shape)
        return out1, out2, out3, out4, out5

class Dual_modal_Aggration_module0_light_end(nn.Module):    
    def __init__(self, in_dim, out_dim): 
        super(Dual_modal_Aggration_module0_light_end, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        

        self.layer_10 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.layer_20 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))

        self.layer_101 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_202 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_301 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.layer_302 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

        self.layer_303 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_304 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
         
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_22 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

        self.layer_33 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)       

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*3, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth):
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)

        x_rgb_s = self.layer_101(rgb)
        x_dep_s = self.layer_202(depth)

        rgb_w = nn.Sigmoid()(x_rgb_s)
        dep_w = nn.Sigmoid()(x_dep_s)

        Rgb_f = x_rgb.mul(dep_w)
        i_f   = x_dep.mul(rgb_w)

        Rgb_f_c1 = self.layer_301(Rgb_f)
        Rgb_f_c2 = self.layer_302(Rgb_f) 

        i_f_c1 = self.layer_303(i_f)
        i_f_c2 = self.layer_304(i_f)

        up_cat   = torch.cat((Rgb_f_c1, i_f_c1),dim=1)
        do_cat   = torch.cat((Rgb_f_c2, i_f_c2),dim=1)

        x_ca_up = self.layer_11(up_cat)
        x_ca_do = self.layer_22(do_cat)

        ou_cat   = torch.cat((x_ca_up, x_ca_do), dim=1)
        out    = self.layer_33(ou_cat)
        return out

class Dual_modal_Aggration_module_light_end(nn.Module):    
    def __init__(self, in_dim, out_dim, m_dim): 
        super(Dual_modal_Aggration_module_light_end, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        
        self.layer_10 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.layer_20 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))

        self.layer_101 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_202 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_301 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.layer_302 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

        self.layer_303 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_304 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
         
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_22 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

        self.layer_33 = nn.Sequential(nn.Conv2d(m_dim+out_dim+out_dim, out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim*2),act_fn)       

        self.layer_44 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth, xx):
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)

        x_rgb_s = self.layer_101(rgb)
        x_dep_s = self.layer_202(depth)

        rgb_w = nn.Sigmoid()(x_rgb_s)
        dep_w = nn.Sigmoid()(x_dep_s)

        Rgb_f = x_rgb.mul(dep_w)
        i_f   = x_dep.mul(rgb_w)

        Rgb_f_c1 = self.layer_301(Rgb_f)
        Rgb_f_c2 = self.layer_302(Rgb_f) 

        i_f_c1 = self.layer_303(i_f)
        i_f_c2 = self.layer_304(i_f)

        up_cat   = torch.cat((Rgb_f_c1, i_f_c1),dim=1)
        do_cat   = torch.cat((Rgb_f_c2, i_f_c2),dim=1)

        x_ca_up = self.layer_11(up_cat)
        x_ca_do = self.layer_22(do_cat)

        ou_cat   = torch.cat((x_ca_up, x_ca_do, xx), dim=1)
        out    = self.layer_44(self.layer_33(ou_cat))
        # out    = self.layer_44(self.layer_33(out))
        return out


