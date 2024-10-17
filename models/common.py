# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

from torch.nn import init, Sequential

import warnings


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)



class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1,  ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out


from torch.autograd import Function

class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input


# class SimAM(torch.nn.Module):
#     def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
#         super(SimAM, self).__init__()
#
#         self.activaton = nn.Sigmoid()
#         self.e_lambda = e_lambda
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         n = w * h - 1
#
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#
#         return x * self.activaton(y)


class Sqrtm(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device)
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
               Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
               Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            ZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
         y = ZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZY, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
         else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                          Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) -
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) -
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) -
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag()
         return grad_input, None
def CovpoolLayer(var):
    return Covpool.apply(var)
def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)
# class SOCA(nn.Module):
#     # second-order Channel attention
#     def __init__(self, channel, reduction=8):
#         super(SOCA, self).__init__()
#         self.max_pool = nn.MaxPool2d(kernel_size=2)
#
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         batch_size, C, h, w = x.shape  # x: NxCxHxW
#         N = int(h * w)
#         min_h = min(h, w)
#         h1 = 1000
#         w1 = 1000
#         if h < h1 and w < w1:
#             x_sub = x
#         elif h < h1 and w > w1:
#             W = (w - w1) // 2
#             x_sub = x[:, :, :, W:(W + w1)]
#         elif w < w1 and h > h1:
#             H = (h - h1) // 2
#             x_sub = x[:, :, H:H + h1, :]
#         else:
#             H = (h - h1) // 2
#             W = (w - w1) // 2
#             x_sub = x[:, :, H:(H + h1), W:(W + w1)]
#         cov_mat = CovpoolLayer(x_sub) # Global Covariance pooling layer
#         cov_mat_sqrt = SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
#         cov_mat_sum = torch.mean(cov_mat_sqrt,1)
#         cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
#         y_cov = self.conv_du(cov_mat_sum)
#         return y_cov*x


def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class SplitAttention(nn.Module):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)
        a = torch.sum(torch.sum(x_all, 1), 1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.reshape(b, self.k, c)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


from collections import OrderedDict

class SKAttention(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V

class SFTLayer_p3(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer_p3, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x, cond):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        return x * (scale + 1) + shift


class SpatialFeatureEnhance_256(nn.Module):

    def __init__(self,arg):
        super(SpatialFeatureEnhance_256, self).__init__()
        self.args=arg
        self.window_size = 7
        self.in_channels = 256
        self.out_channels = 256
        self.inter_channels = 6
        self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
        self.sft = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)

    def forward(self, a):
        b, c, h, w = a[1].shape
        x = a[1].clone()
        a[0] = a[0].permute(0, 2, 3, 1).reshape(-1, c, 1)
        a[1] = F.unfold(a[1], kernel_size=(self.window_size, self.window_size), stride=1,
                            padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                         w).permute(0, 3, 4, 1, 2)
        a[1] = a[1].reshape(-1, c, self.window_size ** 2)

        correlation = torch.bmm(a[0].permute(0, 2, 1), a[1]) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
        sim_map = self.reduce_map(sim_map)
        out = self.sft(x, sim_map)
        return out

class SpatialFeatureEnhance_V1(nn.Module):

        def __init__(self, arg):
            super(SpatialFeatureEnhance_V1, self).__init__()
            self.args = arg
            self.window_size = 7
            self.in_channels = 256
            self.out_channels = 256
            self.inter_channels = 6
            self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
            self.sft_IR = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)
            self.sft_RGB = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)

        def forward(self, a):
            b, c, h, w = a[1].shape
            x = a[1].clone()
            a[0] = a[0].permute(0, 2, 3, 1).reshape(-1, c, 1)
            a[1] = F.unfold(a[1], kernel_size=(self.window_size, self.window_size), stride=1,
                            padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                         w).permute(0, 3, 4, 1, 2)
            a[1] = a[1].reshape(-1, c, self.window_size ** 2)

            correlation = torch.bmm(a[0].permute(0, 2, 1), a[1]) / (c ** 0.5)
            sim_map = F.softmax(correlation, dim=2)
            sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
            sim_map = self.reduce_map(sim_map)
            IR_out = self.sft_IR(x, sim_map)
            RGB_out = self.sft_RGB(sim_map,x)
            return RGB_out,IR_out


class SpatialFeatureEnhance_V2(nn.Module):

    def __init__(self, arg):
        super(SpatialFeatureEnhance_V2, self).__init__()
        self.args = arg
        self.window_size = 7
        self.in_channels = 256
        self.out_channels = 256
        self.inter_channels = 6
        self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
        self.sft_IR = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)
        self.sft_RGB = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)

    def forward(self, a):
        b, c, h, w = a[1].shape
        x = a[1].clone()
        a[0] = a[0].permute(0, 2, 3, 1).reshape(-1, c, 1)
        a[1] = F.unfold(a[1], kernel_size=(self.window_size, self.window_size), stride=1,
                        padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                     w).permute(0, 3, 4, 1, 2)
        a[1] = a[1].reshape(-1, c, self.window_size ** 2)

        correlation = torch.bmm(a[0].permute(0, 2, 1), a[1]) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
        sim_map = self.reduce_map(sim_map)
        IR_out = self.sft_IR(x, sim_map)
        RGB_out = self.sft_RGB(x, sim_map)
        return RGB_out, IR_out
class SpatialFeatureEnhance_V3(nn.Module):

    def __init__(self, arg):
        super(SpatialFeatureEnhance_V3, self).__init__()
        self.args = arg
        self.window_size = 7
        self.in_channels = 256
        self.out_channels = 256
        self.inter_channels = 6
        self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
        self.sft_IR = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)
        self.sft_RGB = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)

    def forward(self, a):
        b, c, h, w = a[1].shape
        x = a[1].clone()
        a[0] = a[0].permute(0, 2, 3, 1).reshape(-1, c, 1)
        a[1] = F.unfold(a[1], kernel_size=(self.window_size, self.window_size), stride=1,
                        padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                     w).permute(0, 3, 4, 1, 2)
        a[1] = a[1].reshape(-1, c, self.window_size ** 2)

        correlation = torch.bmm(a[0].permute(0, 2, 1), a[1]) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
        sim_map = self.reduce_map(sim_map)
        IR_out = self.sft_IR(sim_map,x )
        RGB_out = self.sft_RGB(sim_map,x )
        return RGB_out, IR_out

class SpatialFeatureEnhance_V1_512(nn.Module):

        def __init__(self, arg):
            super(SpatialFeatureEnhance_V1_512, self).__init__()
            self.args = arg
            self.window_size = 7
            self.in_channels = 512
            self.out_channels = 512
            self.inter_channels = 6
            self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
            self.sft_IR = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)
            self.sft_RGB = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)

        def forward(self, a):
            b, c, h, w = a[1].shape
            x = a[1].clone()
            a[0] = a[0].permute(0, 2, 3, 1).reshape(-1, c, 1)
            a[1] = F.unfold(a[1], kernel_size=(self.window_size, self.window_size), stride=1,
                            padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                         w).permute(0, 3, 4, 1, 2)
            a[1] = a[1].reshape(-1, c, self.window_size ** 2)

            correlation = torch.bmm(a[0].permute(0, 2, 1), a[1]) / (c ** 0.5)
            sim_map = F.softmax(correlation, dim=2)
            sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
            sim_map = self.reduce_map(sim_map)
            IR_out = self.sft_IR(x, sim_map)
            RGB_out = self.sft_RGB(sim_map, x)
            return RGB_out, IR_out


class SpatialFeatureEnhance_V1_1024(nn.Module):

    def __init__(self, arg):
        super(SpatialFeatureEnhance_V1_1024, self).__init__()
        self.args = arg
        self.window_size = 7
        self.in_channels = 1024
        self.out_channels = 1024
        self.inter_channels = 6
        self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
        self.sft_IR = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)
        self.sft_RGB = SFTLayer_p3(self.in_channels, self.out_channels, self.inter_channels)

    def forward(self, a):
        b, c, h, w = a[1].shape
        x = a[1].clone()
        a[0] = a[0].permute(0, 2, 3, 1).reshape(-1, c, 1)
        a[1] = F.unfold(a[1], kernel_size=(self.window_size, self.window_size), stride=1,
                        padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                     w).permute(0, 3, 4, 1, 2)
        a[1] = a[1].reshape(-1, c, self.window_size ** 2)

        correlation = torch.bmm(a[0].permute(0, 2, 1), a[1]) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
        sim_map = self.reduce_map(sim_map)
        IR_out = self.sft_IR(x, sim_map)
        RGB_out = self.sft_RGB(sim_map, x)
        return RGB_out, IR_out


#
class SpatialFeatureEnhance_512(nn.Module):

    def __init__(self,arg):
        super(SpatialFeatureEnhance_512, self).__init__()
        self.args=arg
        self.window_size = 7
        self.in_channels = 512
        self.out_channels = 512
        self.inter_channels = 6
        self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
        self.sft = SFTLayer(self.in_channels, self.out_channels, self.inter_channels)

    def forward(self, a):
        b, c, h, w = a[1].shape
        x = a[1].clone()
        a[0] = a[0].permute(0, 2, 3, 1).reshape(-1, c, 1)
        a[1] = F.unfold(a[1], kernel_size=(self.window_size, self.window_size), stride=1,
                            padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                         w).permute(0, 3, 4, 1, 2)
        a[1] = a[1].reshape(-1, c, self.window_size ** 2)

        correlation = torch.bmm(a[0].permute(0, 2, 1), a[1]) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
        sim_map = self.reduce_map(sim_map)
        out = self.sft(x, sim_map)
        return out

class SpatialFeatureEnhance_1024(nn.Module):

    def __init__(self,arg):
        super(SpatialFeatureEnhance_1024, self).__init__()
        self.args=arg
        self.window_size = 7
        self.in_channels = 1024
        self.out_channels = 1024
        self.inter_channels = 6
        self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
        self.sft = SFTLayer(self.in_channels, self.out_channels, self.inter_channels)

    def forward(self, a):
        b, c, h, w = a[1].shape
        x = a[1].clone()
        a[0] = a[0].permute(0, 2, 3, 1).reshape(-1, c, 1)
        a[1] = F.unfold(a[1], kernel_size=(self.window_size, self.window_size), stride=1,
                            padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                         w).permute(0, 3, 4, 1, 2)
        a[1] = a[1].reshape(-1, c, self.window_size ** 2)

        correlation = torch.bmm(a[0].permute(0, 2, 1), a[1]) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
        sim_map = self.reduce_map(sim_map)
        out = self.sft(x, sim_map)
        return out


class SFTLayer(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x, cond):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        return x * (scale + 1) + shift


class SpatialFeatureEnhance(nn.Module):

    def __init__(self, window_size=7, in_channels=3, out_channels=3, inter_channels=6):
        super(SpatialFeatureEnhance, self).__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
        self.sft = SFTLayer(self.in_channels, self.out_channels, self.inter_channels)

    def forward(self, query_data, key_data):
        b, c, h, w = key_data.shape
        x = key_data.clone()
        query_data = query_data.permute(0, 2, 3, 1).reshape(-1, c, 1)
        key_data = F.unfold(key_data, kernel_size=(self.window_size, self.window_size), stride=1,
                            padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                         w).permute(0, 3, 4, 1, 2)
        key_data = key_data.reshape(-1, c, self.window_size ** 2)

        correlation = torch.bmm(query_data.permute(0, 2, 1), key_data) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
        sim_map = self.reduce_map(sim_map)
        out = self.sft(x, sim_map)
        return out

class NonLocalBlockND_SFE(nn.Module):
    def __init__(self, in_channels=256, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND_SFE, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.sfe=SpatialFeatureEnhance(window_size=7, in_channels=256, out_channels=256, inter_channels=6)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        # z = W_y + x
        z=self.sfe(W_y,x)
        return z


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

# CA
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         mip = max(8, inp // reduction)
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#         n, c, h, w = x.size()
#         # c*1*W
#         x_h = self.pool_h(x)
#         # c*H*1
#         # C*1*h
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#         y = torch.cat([x_h, x_w], dim=2)
#         # C*1*(h+w)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#         out = identity * a_w * a_h

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class CB2d(nn.Module):
    def __init__(self, inplanes, pool='att', fusions=['channel_add', 'channel_mul']):
        super(CB2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = inplanes // 4
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':  # iscyy
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(3)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out

# class DilatedContextAttentionModule(nn.Module):
#     def __init__(self, mode='dot_product', channels=256):
#         super(DilatedContextAttentionModule, self).__init__()
#         self.in_channels = channels
#         self.inter_channels = channels
#         self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                            kernel_size=1, stride=1, padding=0)
#
#         self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
#                            kernel_size=1, stride=1, padding=0)
#
#         self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                                kernel_size=1, stride=1, padding=0)
#         self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                              kernel_size=1, stride=1, padding=0)
#
#         self.bn = nn.BatchNorm2d(self.in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
#         if mode == 'dot_product':
#             self.operation_function = self._dot_product
#         else:
#             print("not implement")
#
#     def _dot_product(self, xi, xj):
#         batch_size = xi.size(0)
#         # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
#         g_x = self.g(xj).view(batch_size, self.inter_channels, -1)
#         g_x = g_x.permute(0, 2, 1)
#
#         theta_x = self.theta(xi).view(batch_size, self.inter_channels, -1)
#         theta_x = theta_x.permute(0, 2, 1)
#         phi_x = self.phi(xj).view(batch_size, self.inter_channels, -1)
#
#         f = torch.matmul(theta_x, phi_x)
#         N = f.size(-1)
#         f_div_C = f / N
#
#         y = torch.matmul(f_div_C, g_x)
#         y = y.permute(0, 2, 1).contiguous()
#         y = y.view(batch_size, self.inter_channels, *xi.size()[2:])
#         W_y = self.W(y)
#         z = W_y + xi
#         z = self.bn(z)
#         return z
#
#     def forward(self, xi, xj):
#         '''
#         :param xi, xj: (b, c, h, w)
#         :return: (b, c, h, w)
#         '''
#         output = self.operation_function(xi, xj)
#         return output
#
#
#
#
# class ContextAttentionModule(nn.Module):
#
#     def __init__(self, input_channels=3,padding_dilation=2 ):
#         super(ContextAttentionModule, self).__init__()
#         self.dilation = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=padding_dilation,
#                                   dilation=padding_dilation, bias=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.dcam = DilatedContextAttentionModule(mode='dot_product', channels=input_channels)
#         self.wr = nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, stride=1, padding=0, bias=True)
#
#     def forward(self, pn):
#         pn_dilated = self.dilation(pn)
#         pn_dilated = self.relu(pn_dilated)
#         pn_dcam = self.dcam(pn_dilated, pn)
#         concat = torch.cat((pn_dcam, pn), 1)
#         out = self.wr(concat)
#         return out

class SpaceToDepth(nn.Module):
    def __init__(self, block_size=2):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


import torch
from torch import nn
import math
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    # 构建DCT滤波器，对应数学公式
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        # 返回一组DCT滤波器
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    # 构建DCT滤波器，对应数学公式
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        # DCT变换
        # x与DCT滤波器内积
        x = x * self.weight
        # 消去x的2,3维
        result = torch.sum(x, dim=[2, 3])
        return result

class NonLocalBlockND2(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND2, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x,y):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(y).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NONLocalBlock2D(NonLocalBlockND2):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)




class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel=256, dct_h=80, dct_w=80, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        # 返回x的多光谱向量
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        # 全连接层 + relu激活 + 全连接层 + sigmoid激活，返回一个通道注意力向量
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        # 通道注意力向量与原图像x相乘
        return x * y.expand_as(x)
class DilatedContextAttentionModule(nn.Module):
    def __init__(self, mode='dot_product', channels=256):
        super(DilatedContextAttentionModule, self).__init__()
        self.in_channels = channels
        self.inter_channels = channels
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(self.in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        if mode == 'dot_product':
            self.operation_function = self._dot_product
        else:
            print("not implement")

    def _dot_product(self, xi, xj):
        batch_size = xi.size(0)
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(xj).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(xi).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(xj).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *xi.size()[2:])
        W_y = self.W(y)
        z = W_y + xi
        z = self.bn(z)
        return z

    def forward(self, xi, xj):
        '''
        :param xi, xj: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        output = self.operation_function(xi, xj)
        return output


class ContextAttentionModule(nn.Module):

    def __init__(self, input_channels=3):
        super(ContextAttentionModule, self).__init__()
        self.dcam = DilatedContextAttentionModule(mode='dot_product', channels=input_channels)
        self.wr = nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, pn,pm):

        pn_dcam = self.dcam(pn, pm)
        concat = torch.cat((pn, pm), 1)
        out = self.wr(concat)
        return out

class CFEL(nn.Module):
    def __init__(self,outc):
        super(CFEL,self).__init__()
        self.outc = outc
        self.space2depth=SpaceToDepth()
        self.cbam=CBAM(256)
        self.conv=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1)
        self.non_local= NONLocalBlock2D(256, sub_sample=True, bn_layer=True)
    def forward(self,x):
        space=self.space2depth(x[0])
        con=self.conv(space)
        relu=torch.relu(con)
        cbam=self.cbam(relu)
        res=self.non_local(cbam,x[1])
        return res


class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class SimAM_conv_256(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM_conv_256, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam_conv_256"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        ww=self.conv(self.activaton(y))
        return x * ww

class SimAM_conv_512(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM_conv_512, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam_conv_512"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        ww=self.conv(self.activaton(y))
        return x * ww
class SimAM_conv_1024(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM_conv_1024, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.conv = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam_conv_1024"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        ww=self.conv(self.activaton(y))
        return x * ww



class C3GC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion #iscyy

        super(C3GC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.gc = CB2d(c1)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        # self.m = nn.Sequential(*[CB2d(c_) for _ in range(n)])
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        out = torch.cat((self.m(self.cv1(self.gc(x))), self.cv2(self.gc(x))), dim=1)
        out = self.cv3(out)
        return out

class GCBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.gc = CB2d(c1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.gc(self.cv1(x))) if self.add else self.cv2(self.gc(self.cv1(x)))
        # 注意到这里的self.gc有两个，就分别对应于C3GC结构图的两个分支，如果想要只融合一个分支便删掉一个self.gc即可。



class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])



class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out


class Add_gpt_0(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add_gpt_0, self).__init__()
        self.arg = arg
        self.GPT_0 = SpatialFeatureEnhance_gpt(256)
    def forward(self, x):
        return self.GPT_0(x[1][0],x[0])
class Add_gpt_1(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add_gpt_1, self).__init__()
        self.arg = arg
        self.GPT_1=SpatialFeatureEnhance_gpt(256)
    def forward(self, x):
        return self.GPT_1(x[1][1],x[0])



class SpatialFeatureEnhance_gpt(nn.Module):

    def __init__(self, arg):
        super(SpatialFeatureEnhance_gpt, self).__init__()
        self.window_size = 7
        self.in_channels = 256
        self.out_channels = 256
        self.inter_channels = 6
        self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
        self.sft = SFTLayer(self.in_channels, self.out_channels, self.inter_channels)

    def forward(self, query_data, key_data):
        b, c, h, w = key_data.shape
        x = key_data.clone()
        query_data = query_data.permute(0, 2, 3, 1).reshape(-1, c, 1)
        key_data = F.unfold(key_data, kernel_size=(self.window_size, self.window_size), stride=1,
                            padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                         w).permute(0, 3, 4, 1, 2)
        key_data = key_data.reshape(-1, c, self.window_size ** 2)

        correlation = torch.bmm(query_data.permute(0, 2, 1), key_data) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size ** 2).permute(0, 3, 1, 2)
        sim_map = self.reduce_map(sim_map)
        out = self.sft(x, sim_map)
        return out


class SFTLayer_v1(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer_v1, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.simam=SimAM()
    def forward(self, x, cond):
        # x[0]: fea; x[1]: cond
        cond=self.simam(cond)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        return x * (scale + 1) + shift


class SFTLayer_v2(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer_v1, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.simam_scale=SimAM()
        self.simam_shift = SimAM()
    def forward(self, x, cond):
        # x[0]: fea; x[1]: cond

        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        scale=self.simam_scale(scale)
        shift=self.simam_shift(shift)
        return x * (scale + 1) + shift

class SpatialFeatureEnhance_V1_sftv1(nn.Module):

        def __init__(self, arg):
            super(SpatialFeatureEnhance_V1_sftv1, self).__init__()
            self.args = arg
            self.window_size = 7
            self.in_channels = 256
            self.out_channels = 256
            self.inter_channels = 6
            self.reduce_map = nn.Conv2d(self.window_size ** 2, self.in_channels, 1)
            self.sft_IR = SFTLayer_v1(self.in_channels, self.out_channels, self.inter_channels)
            self.sft_RGB = SFTLayer_v1(self.in_channels, self.out_channels, self.inter_channels)

        def forward(self, a):
            b, c, h, w = a[1].shape
            x = a[1].clone()

            a[1] = F.unfold(a[1], kernel_size=(self.window_size, self.window_size), stride=1,
                            padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size ** 2, h,
                                                                                         w).permute(0, 3, 4, 1, 2)


            correlation = torch.bmm(a[0].permute(0, 2, 1), a[1]) / (c ** 0.5)
            sim_map = F.softmax(correlation, dim=2)

            sim_map = self.reduce_map(sim_map)
            IR_out = self.sft_IR(x, sim_map)
            RGB_out = self.sft_RGB(sim_map, x)
            return RGB_out, IR_out


