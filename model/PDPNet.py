import sys
#from Unet_modified import conv_block
sys.path.append('./')
sys.path.append('../')
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def gen_grid2d(h, w, device = 'cuda'):
    x_grid = torch.arange(0, h).unsqueeze(1).expand(-1, w)
    y_grid = torch.arange(0, w).unsqueeze(0).expand(h, -1)
    x_grid = x_grid.to(device)
    y_grid = y_grid.to(device)
    return x_grid, y_grid

def first_order_moment2d(data, device = 'cuda'):
    n, c, h, w = np.shape(data)
    moment_00 = torch.sum(data, dim = [2, 3])
    x_grid, y_grid = gen_grid2d(h, w, device)
    
    moment_10 = torch.sum(data * x_grid.unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1).expand(-1, c, -1, -1), dim = [2, 3])
    moment_01 = torch.sum(data * y_grid.unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1).expand(-1, c, -1, -1), dim = [2, 3])
    
    return moment_00, moment_10, moment_01

def get_center_2d(data, device):
    moment_00, moment_10, moment_01 = first_order_moment2d(data, device)
    X_center = moment_10 / (moment_00 + 1e-4)
    Y_center = moment_01 / (moment_00 + 1e-4)
    return X_center.int(), Y_center.int()

def get_patch_2d(data, x_centers, y_centers, patchsize, device = 'cuda'):
    n, c = x_centers.size()
    _, _, h, w = data.size()

    x_offset = torch.arange(-(patchsize[0] // 2), (patchsize[0] // 2)).unsqueeze(0).unsqueeze(0).expand(n, -1, -1).expand(-1, c, -1)
    y_offset = torch.arange(-(patchsize[1] // 2), (patchsize[1] // 2)).unsqueeze(0).unsqueeze(0).expand(n, -1, -1).expand(-1, c, -1)
    
    x_offset = x_offset.to(device)
    y_offset = y_offset.to(device)
    
    x_low_bound = x_centers - patchsize[0] // 2 
    x_low_shift = (0 - x_low_bound) * (x_low_bound < 0).float()
    x_centers = x_centers + x_low_shift
    x_up_bound = x_centers + patchsize[0] // 2
    x_up_shift = (h - x_up_bound) * (x_up_bound > h).float()
    x_centers = x_centers + x_up_shift

    y_low_bound = y_centers - patchsize[1] // 2
    y_low_shift = (0 - y_low_bound) * (y_low_bound <= 0).float()
    y_centers = y_centers + y_low_shift
    y_up_bound = y_centers + patchsize[1] // 2
    y_up_shift = (w - y_up_bound) * (y_up_bound > w).float()
    y_centers = y_centers + y_up_shift

    x_patch = x_centers.unsqueeze(-1).expand(-1, -1, x_offset.size(-1)) + x_offset
    y_patch = y_centers.unsqueeze(-1).expand(-1, -1, y_offset.size(-1)) + y_offset
    tpatches = [] 
    for bi in range(n):
        tpatch = torch.index_select(data[bi : bi + 1, :, ...], 2, x_patch[bi, 0, :].long(), out=None)
        tpatch = torch.index_select(tpatch, 3, y_patch[bi, 0, :].long(), out=None)
        tpatches.append(tpatch)
    tpatches = torch.cat(tpatches, 0)
    
    return tpatches

class CeLossSigmoid(nn.Module):
    def __init__(self):
        super(CeLossSigmoid, self).__init__()
        self.epsilon = 1e-5
        
    def forward(self, predict, target):
        pre = predict.view(predict.size(0), -1)      
        tar = target.view(target.size(0), -1)
        ce = (tar * torch.log(pre + self.epsilon) + (1 - tar) * torch.log(1 - pre + self.epsilon)) * (-1)
        return ce.mean(-1).mean()

class LogCoshDiceSigmoid(nn.Module):
    def __init__(self):
        super(LogCoshDiceSigmoid, self).__init__()
        self.epsilon = 1e-5
        
    def forward(self, predict, target):
        pre = predict.view(predict.size(0), predict.size(1), -1)      
        tar = target.view(target.size(0), target.size(1), -1)
               
        score = 1 - 2 * (pre * tar + (1 - pre) * (1 - tar)).sum(-1) / ((pre + tar) + ((1 - pre) + (1 - tar))).sum(-1)
        return torch.log((torch.exp(score.mean()) + torch.exp(score.mean())) / 2.0)

from model import DenseNet5s as DenseNet
from model import DPKNet
class PDPNet(nn.Module):
    def __init__(self, 
                 locmodel = DenseNet.densenet121(),
                 segmodel = DPKNet.DPKNet()
                 ):
        super(PDPNet, self).__init__()
        self.locationEncoder = locmodel
        self.locationLogit = nn.Sequential(
                    nn.Conv2d(1024, 1, 1, 1, 0),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
            )
        self.sefNet = segmodel
        self.dice_loss = LogCoshDiceSigmoid()
        self.ce_loss = CeLossSigmoid()
        self.sizelist = [64, 80, 96, 112, 128]
                    
    def forward(self, x, y):
        _, _, _, _, lm5 = self.locationEncoder(x)
        lL = self.locationLogit(lm5)
                
        with torch.no_grad():
            mask_thres = F.adaptive_avg_pool2d(y.float(), [1, 1])
            label = F.avg_pool2d(y.float(), 
                                    [y.size(2) // lL.size(2), y.size(3) // lL.size(3)], 
                                    [y.size(2) // lL.size(2), y.size(3) // lL.size(3)])
            label = (label > mask_thres).long()
        ldice = self.ce_loss(lL, label)
            
        with torch.no_grad():
            y_np = F.interpolate((lL > 0.5).float(), 
                                 [x.size(2), x.size(3)], 
                                 mode = 'bicubic', 
                                 align_corners = True).detach().cpu().numpy()
            cropxs = []
            cropys = []
            x_center = []
            y_center = []
            patchsize = []
            for j in range(y.size(0)):
                x_position, y_position = np.nonzero(y_np[j, 0])
                if len(x_position) != 0:
                    x_min, x_max = np.min(x_position), np.max(x_position)
                    tx_center = x_min + (x_max - x_min) // 2
                    y_min, y_max = np.min(y_position), np.max(y_position)
                    ty_center = y_min + (y_max - y_min) // 2
                    
                    tpatchsize = np.maximum(x_max - x_min + 1, y_max - y_min + 1)
                    tpatchsize = int((128 - tpatchsize) * 0.5) + tpatchsize
                    # tpatchsize = np.maximum(tpatchsize, 64)
                    
                else:
                    x_min = 0
                    x_max = 127
                    y_min = 0
                    y_max = 127
                    
                    tx_center = x_min + (x_max - x_min) // 2
                    ty_center = y_min + (y_max - y_min) // 2
                    
                    tpatchsize = 128
                
                x_center.append(tx_center)
                y_center.append(ty_center)
                patchsize.append(tpatchsize)
            
            for j in range(y.size(0)):
                if x_center[j] + patchsize[j] // 2 > x.size(2):
                    x_center[j] = x_center[j] - (x_center[j] + patchsize[j] // 2 - x.size(2))
                elif x_center[j] - patchsize[j] // 2 < 0:
                    x_center[j] = x_center[j] + (patchsize[j] // 2 - x_center[j])
            
                if y_center[j] + patchsize[j] // 2 > x.size(3):
                    y_center[j] = y_center[j] - (y_center[j] + patchsize[j] // 2 - x.size(3))
                elif y_center[j] - patchsize[j] // 2 < 0:
                    y_center[j] = y_center[j] + (patchsize[j] // 2 - y_center[j])
                    
                cropx = x[j: j + 1, :, 
                          x_center[j] - patchsize[j] // 2: x_center[j] + patchsize[j] // 2, 
                          y_center[j] - patchsize[j] // 2: y_center[j] + patchsize[j] // 2]
                cropx = (cropx - torch.min(cropx)) / (torch.max(cropx) - torch.min(cropx))
                cropx = F.interpolate(cropx, (x.size(2), x.size(3)), mode = 'bicubic', align_corners = True)
                cropxs.append(cropx)
                
                cropy = y[j: j + 1, :, x_min: x_max, y_min: y_max]
                cropy = F.interpolate(cropy.float(), (x.size(2), x.size(3)), mode = 'bilinear', align_corners = True)
                cropy = (cropy > 0).long()
                cropys.append(cropy)
            cropxs = torch.cat(cropxs, 0)
            cropys = torch.cat(cropys, 0)
            
        pres = self.sefNet(cropxs)
        cdice = 0
        mask_thres = F.adaptive_avg_pool2d(cropys.float(), [1, 1])
        for i, L in enumerate(pres):
            label = F.avg_pool2d(cropys.float(), 
                                 [cropys.size(2) // L.size(2), cropys.size(3) // L.size(3)], 
                                 [cropys.size(2) // L.size(2), cropys.size(3) // L.size(3)])
            label = (label > mask_thres).long()
            if i == 0:
                cdice = self.ce_loss(L, label)
            else:
                cdice = cdice + self.dice_loss(L, label)
        
        with torch.no_grad():
            xs = torch.zeros_like(x)
            ls = torch.zeros_like(pres[-1])
            ys = torch.zeros_like(y)
            cropys = cropys.float()
            for j in range(y.size(0)):
                xs[j: j + 1, :, 
                   x_center[j] - patchsize[j] // 2: x_center[j] + patchsize[j] // 2, 
                   y_center[j] - patchsize[j] // 2: y_center[j] + patchsize[j] // 2] = F.interpolate(cropxs[j:j + 1, :, :, :], 
                                                                           (patchsize[j] // 2 * 2, patchsize[j] // 2 * 2), 
                                                                           mode = 'bicubic', 
                                                                           align_corners = True)
                ls[j: j + 1, :, 
                   x_center[j] - patchsize[j] // 2: x_center[j] + patchsize[j] // 2, 
                   y_center[j] - patchsize[j] // 2: y_center[j] + patchsize[j] // 2] = F.interpolate(pres[-1][j:j + 1, :, :, :], 
                                                                           (patchsize[j] // 2 * 2, patchsize[j] // 2 * 2), 
                                                                           mode = 'bilinear', 
                                                                           align_corners = True)
                ys[j: j + 1, :, 
                   x_center[j] - patchsize[j] // 2: x_center[j] + patchsize[j] // 2, 
                   y_center[j] - patchsize[j] // 2: y_center[j] + patchsize[j] // 2] = F.interpolate(cropys[j:j + 1, :, :, :], 
                                                                           (patchsize[j] // 2 * 2, patchsize[j] // 2 * 2), 
                                                                           mode = 'bilinear', 
                                                                           align_corners = True)
            ys = (ys > 0).long()
        return [lL, ls], [xs, ys], [ldice, cdice]

def debug():
    data = np.random.rand(1 * 1 * 128 *128)
    data = np.reshape(data, [1, 1, 128, 128])
    data = torch.tensor(data, dtype = torch.float32, device = 'cuda')
    model = PDPNet()
    model.cuda()
    model.eval()
    logit, x, loss = model(data, data)
    print(logit[0].size(), logit[1].size())
    print(x[0].size(), x[1].size())
    print(loss[0], loss[1])
    
if __name__ == '__main__':
    debug()
#     GassianBlurConvdebug()