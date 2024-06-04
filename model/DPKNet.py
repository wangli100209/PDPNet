import sys

#from Unet_modified import conv_block
sys.path.append('./')
sys.path.append('../')
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from model import DenseNet5s as DenseNet

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

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, kernel_size=3,stride=1,padding=1):
        super(up_conv,self).__init__()
        self.inter = nn.Upsample(scale_factor=2)
        self.conv = nn.Sequential(
                    # PSCModule(ch_in, ch_out, (1, 3, 5, 7)),
                    nn.Conv2d(ch_in,ch_out, kernel_size,stride,padding),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True)
            )
        # self.reduction = conv2d_block(ch_in + ch_out, ch_out, 1, 1, 0)
                    
    def forward(self,x):
        x = self.inter(x)
        x = self.conv(x)
        # out = torch.cat([x, psc], 1)
        return x

class conv2d_block(nn.Module):
    def __init__(self,ch_in,ch_out, kernel_size, stride, padding):
        super(conv2d_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class conv2dx2_block(nn.Module):
    def __init__(self,ch_in,ch_out, kernel_size, stride, padding):
        super(conv2dx2_block,self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=stride, padding=padding,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )

    def forward(self,x):
        x = self.conv0(x)
        x = self.conv1(x)
        return x

class MaskCrossScaleAttention(nn.Module):
    def __init__(self, ch_en, ch_de, ch_out):
        super(MaskCrossScaleAttention, self).__init__()
        
        self.en_foward = nn.Sequential(
                                        conv2d_block(ch_en, ch_en // 4, 3, 1, 1),
                                        nn.MaxPool2d(3, 2, 1),
                                        nn.Conv2d(ch_en // 4, ch_en // 4, 1, 1, 0,
                                                  groups = ch_en // 4)
                                        )
        self.de_foward = nn.Sequential(
                                        nn.Conv2d(ch_de, ch_de // 4, 1, 1, 0)
                                        )
        
        self.query_proj = nn.Conv2d(ch_en // 4 + ch_de // 4,
                                    ch_en + ch_en, 1, 1, 0
                                    )
        self.key_proj = nn.Conv2d(ch_en // 4 + ch_de // 4,
                                  ch_en + ch_en, 1, 1, 0
                                  )
        self.value_proj = nn.Conv2d(ch_en // 4 + ch_de // 4,
                                    ch_en + ch_en, 1, 1, 0
                                    )
        
        self.bottleneck = conv2d_block(ch_en + ch_en, ch_out, 1, 1, 0)
        
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(-1)

    def forward(self, em, dm, p):
        em_f = self.en_foward(em)
        dm_f = self.de_foward(dm)
        
        f = torch.cat([em_f, dm_f], 1)
        fb, fc, fh, fw = f.size()
        
        f_q = self.query_proj(f).view(fb, -1, fh * fw).permute(0, 2, 1)
        f_k = self.key_proj(f).view(fb, -1, fh * fw)
        
        energy = torch.bmm(f_q, f_k)
        self_att = self.sigmoid(energy)
        
        f_v = self.value_proj(f)
        
        p_v = f_v * p
        
        self_v = torch.bmm( f_v.view(fb, -1, fh * fw), self_att.permute(0,2,1) )
        
        o_v = ( p_v + self_v.reshape( p_v.size() ) )
        o_v = self.bottleneck(o_v)
        return o_v
    
    def forward_feature(self, em, dm ,p):
        em_f = self.en_foward(em)
        dm_f = self.de_foward(dm)
        
        f = torch.cat([em_f, dm_f], 1)
        fb, fc, fh, fw = f.size()
        
        f_q = self.query_proj(f).view(fb, -1, fh * fw).permute(0, 2, 1)
        f_k = self.key_proj(f).view(fb, -1, fh * fw)
        
        energy = torch.bmm(f_q, f_k)
        self_att = self.sigmoid(energy)
        
        f_v = self.value_proj(f)
        
        p_v = f_v * p
        
        self_v = torch.bmm( f_v.view(fb, -1, fh * fw), self_att.permute(0,2,1) )
        
        s_mask = self_att.sum(1)
        o_mask = torch.cat([self_att, p.view(fb, 1, -1)], 1) 
        o_mask = o_mask.sum(1)
        
        o_v = ( p_v + self_v.reshape( p_v.size() ) )
        o_v = self.bottleneck(o_v)
        
        s_mask = s_mask.reshape(p.size())
        tmin = s_mask.min(2, True)[0].min(3, True)[0]
        tmax = s_mask.max(2, True)[0].max(3, True)[0]
        s_mask = ( s_mask - tmin ) / (tmax - tmin)
        
        o_mask = o_mask.reshape(p.size())
        tmin = o_mask.min(2, True)[0].min(3, True)[0]
        tmax = o_mask.max(2, True)[0].max(3, True)[0]
        o_mask = ( o_mask - tmin ) / (tmax - tmin)
        
        return o_v, s_mask, o_mask

class DPKNet(nn.Module):
    def __init__(self, channels=1):
        super(DPKNet, self).__init__()       
        kwargs = {"ch_in": channels} 
        self.num_features = 1024
        self.encoder = DenseNet.densenet121(**kwargs)
        
        self.att5 = MaskCrossScaleAttention(256, 1024, 256)
        self.att4 = MaskCrossScaleAttention(128, 256, 128)
        self.att3 = MaskCrossScaleAttention(64, 128, 64)
        
        self.logit5 = nn.Sequential(
                    nn.Conv2d(1024, 1, 1, 1, 0),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
            )
        self.logit4 = nn.Sequential(
                    nn.Conv2d(256, 1, 1, 1, 0),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
            )
        self.logit3 = nn.Sequential(
                    nn.Conv2d(128, 1, 1, 1, 0),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
            )
        self.logit0 = nn.Sequential(
                    nn.Conv2d(16, 1, 1, 1, 0),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
            )
        
        self.conv4 = conv2dx2_block(512, 256, 3, 1, 1)
        self.conv3 = conv2dx2_block(256, 128, 3, 1, 1)
        self.conv2 = conv2dx2_block(128, 64, 3, 1, 1)
        
        self.upconv1 = up_conv(64, 64, 3, 1, 1)
        
        self.conv1 = nn.Sequential(
                    nn.Conv2d(128, 32, 3, 1, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
            )
        
        self.upconv0 = up_conv(32, 16, 3, 1, 1)
        
        
    def forward(self, x):        
        m1, m2, m3, m4, m5 = self.encoder(x)
        L5 = self.logit5(m5)
        
        M4 = self.att5(m4, m5, L5)
        M4 = torch.cat([F.interpolate( M4, [m4.size(2), m4.size(3)] ), m4], 1)
        M4 = self.conv4(M4)
        L4 = self.logit4(M4)
        
        M3 = self.att4(m3, M4, L4)
        M3 = torch.cat([F.interpolate( M3, [m3.size(2), m3.size(3)] ), m3], 1)
        M3 = self.conv3(M3)
        L3 = self.logit3(M3)
        
        M2 = self.att3(m2, M3, L3)
        M2 = torch.cat([F.interpolate( M2, [m2.size(2), m2.size(3)] ), m2], 1)
        M2 = self.conv2(M2)

        M1 = self.upconv1(M2)
        M1 = self.conv1(torch.cat([M1, m1], 1))
        
        M0 = self.upconv0(M1)
        L0 = self.logit0(M0)
        
        return L5, L4, L3, L0

def debug_model():
    data = torch.rand([4, 1, 128, 128])
    
    model_list = []
    model_list.append( DPKNet(1) )
        
    for model in model_list:
        pres = model(data)
        print(pres[-1].size())

if __name__ == '__main__':
    # debug_module()
    debug_model()