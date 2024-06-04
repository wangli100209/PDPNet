import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def GaussianKernel(mode = '2d',kernel_size=3,sigma=1):
    if mode=='2d':
        h=w=(kernel_size-1.)/2.
        y, x= np.ogrid[-h:h+1,-w:w+1]
        h=np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh=h.sum()
        if sumh!=0:
            h/=sumh
        return h
    else:
        h=w=d=(kernel_size - 1.) / 2.
        y, x, z= np.ogrid[-h:h+1,-w:w+1, -d:d+1]
        h=np.exp( -(x*x + y*y + z*z) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh=h.sum()
        if sumh!=0:
            h/=sumh
        return h
    
class GaussianLayer(nn.Module):
    def __init__(self, mode = '3d', channels=3, kernelsize = 3, sigma = 1):
        super(GaussianLayer, self).__init__()
        self.channels = channels
        kernel = GaussianKernel(mode, kernelsize, sigma)       
        self.kernelsize = kernelsize
        self.mode = mode
        
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        if mode == '2d':
            kernel = kernel.expand(-1, self.channels, -1, -1)
        else:
            kernel = kernel.expand(-1, self.channels, -1, -1, -1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
    def forward(self, x):
        with torch.no_grad():
            if self.mode == '2d':
                x = F.conv2d(x, self.weight, padding=self.kernelsize // 2)
            else:
                x = F.conv3d(x, self.weight, padding=self.kernelsize // 2)
        return x

def gen_gradkernel(mode = '3d'):
    if mode == '2d':
        dir = np.array([1, 0, 1])
        w = np.array([1, 2, 1])
        x_kernel = np.einsum("i, j->ij", w, dir)
        x_kernel[:, 0] = x_kernel[:, 0] * -1     
        y_kernel = np.einsum("i, j->ji", w, dir)
        y_kernel[-1] = y_kernel[-1] * -1
        return np.concatenate([x_kernel[np.newaxis, ...], y_kernel[np.newaxis, ...]], 0)
    else:
        dir = np.array([1, 0, 1])
        w = np.array((
                [[1, 2, 1],
                 [1, 2, 1],
                 [1, 2, 1]
                 ]
            ))
        z_kernel = np.einsum("ij, k->ijk", w, dir)
        z_kernel[:, :, 0] = z_kernel[:, :, 0] * -1
        x_kernel = np.einsum("ij, k->ikj", w, dir)
        x_kernel[:, 0, :] = x_kernel[:, 0, :] * -1
        y_kernel = np.einsum("ij, k->kji", w, dir)
        y_kernel[0, :, :] = y_kernel[0, :, :] * -1
        return np.concatenate([x_kernel[np.newaxis, ...], y_kernel[np.newaxis, ...], z_kernel[np.newaxis, ...]], 0)

class SobelLayer(nn.Module):
    def __init__(self, mode = '2d', channels=3):
        super(SobelLayer, self).__init__()
        self.mode = mode
        self.channels = channels
        kernels = gen_gradkernel(mode)        
        kernels = torch.FloatTensor(kernels).unsqueeze(1)
        if mode == '2d':
            kernel = kernels.expand(-1, self.channels, -1, -1)
        else:
            kernel = kernels.expand(-1, self.channels, -1, -1, -1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
    def forward(self, x):
        with torch.no_grad():
            if self.mode == '2d':
                x = F.conv2d(x, self.weight, padding=1)
            else:
                x = F.conv3d(x, self.weight, padding=1)
        return x