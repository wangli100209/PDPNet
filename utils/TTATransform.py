import sys
sys.path.append('./')
sys.path.append('../')
import numpy as np
from torch.nn.modules import padding
from utils import Utils
from torchvision import transforms
from torchvision.transforms import functional as TrsF
from torchvision.transforms import InterpolationMode
from PIL import Image
import torch
from torch.nn import functional as F

def getSize(item):
    if isinstance(item, Image.Image):
        return item.shape
    elif isinstance(item, torch.Tensor):
        return item.size()[2:]

class ResizeFuc(torch.nn.Module):
    def __init__(self,  size, interpolation = InterpolationMode.NEAREST):
        super().__init__()
        self.name = 'resize'
        self.paras = {'size': size, 'interpolation': interpolation}
        self.tempparams = {'interpolation': interpolation}

    def forward(self, itemlist, *args):
        tdatalist = []
        for item in itemlist:
            self.tempparams.update( { 'size': getSize(item) } )
            tdatalist.append(TrsF.resize(item, **self.paras))
        return tdatalist
    
    def invforward(self, itemlist, *args):
        tdatalist = []
        for item in itemlist:
            tdatalist.append(TrsF.resize(item, **self.tempparams))
        return tdatalist
    
class ResizeCropFuc(torch.nn.Module):
    def __init__(self,  size, scale = (0.08, 1.0), ratio = (3. / 4., 4. / 3.), interpolation = InterpolationMode.NEAREST):
        super().__init__()
        self.name = 'resizeCrop'
        self.paras = {'size': size, 'scale': scale, 'ratio': ratio, 'interpolation': interpolation}
        self.tempparams = {}
        self.funcClass = transforms.RandomResizedCrop(**self.paras)

    def forward(self, itemlist, *args):
        Paras = self.funcClass.get_params(itemlist[0], self.funcClass.scale, self.funcClass.ratio)
        self.tempparams.update(
            {
                'i': Paras[0],
                'j': Paras[1],
                'h': Paras[2],
                'w': Paras[3]
            }
        )
        tdatalist = []
        for item in itemlist:
            self.tempparams.update( { 'size': getSize(item) } )
            item = TrsF.crop(item, *Paras)
            item = TrsF.resize(item, self.paras['size'], self.paras['interpolation'])
            tdatalist.append(item)
        return tdatalist
    
    # torch padding  [left, right, top, bottom]
    def invforward(self, itemlist, *args):
        i, j, h, w = self.tempparams['i'], self.tempparams['j'], self.tempparams['h'], self.tempparams['w']
        size = self.tempparams['size']
        top_ind, bottom_ind = i,  i + h
        left_ind, right_ind = j, j + w
        if isinstance(itemlist[0], torch.Tensor):
            left_pad, right_pad = left_ind, size[1] - right_ind
            top_pad, bottom_pad = top_ind, size[0] - bottom_ind
        tdatalist = []
        for item in itemlist:
            item = TrsF.resize(item, [h, w], self.paras['interpolation'])
            if isinstance(item, torch.Tensor):
                tdatalist.append( F.pad(item, [left_pad, right_pad, top_pad, bottom_pad]) )
            else:
                tdata = np.zeros(size)
                tdata[top_ind: bottom_ind, left_ind: right_ind] = np.array(item)
                tdatalist.append( Image.fromarray(tdata) )
        return tdatalist

class AffineFunc(torch.nn.Module):
    def __init__(self,  degrees, translate=None, scale=None, shear=None, interpolation=InterpolationMode.NEAREST, fill=0,
        fillcolor=None, resample=None):
        super().__init__()
        self.name = 'affine'
        self.paras = {'degrees': degrees, 'translate': translate, 'scale': scale, 'shear' : shear, 'interpolation': InterpolationMode.BICUBIC,
                'fill': fill
             }
        self.tempparams = {}
        self.funcClass = transforms.RandomAffine(**self.paras)

    def forward(self, itemlist, *args):
        Paras = self.funcClass.get_params(self.funcClass.degrees, 
                                            self.funcClass.translate, 
                                            self.funcClass.scale, 
                                            self.funcClass.shear, getSize(itemlist[0]))
        self.tempparams.update({
            'angle': Paras[0], 
            'translations': Paras[1], 
            'scale': Paras[2], 
            'shear': Paras[3]
        })
        tdatalist = []
        for item in itemlist:
            item = TrsF.affine(item, *Paras)
            tdatalist.append(item)
        return tdatalist
    
    def invforward(self, itemlist, *args):
        angle = self.tempparams['angle'] * -1
        translations = list(self.tempparams['translations'])
        translations[0] = translations[0] * -1
        translations[1] = translations[1] * -1
        scale = 1 / self.tempparams['scale']
        shear = list(self.tempparams['shear'])
        shear[0] = shear[0] * -1
        shear[1] = shear[1] * -1
        tdatalist = []
        for item in itemlist:
            item = TrsF.affine(item, 0, tuple(translations), 1, tuple([0, 0]))
            item = TrsF.affine(item, 0, tuple([0, 0]), 1, tuple(shear))
            item = TrsF.affine(item, 0, tuple([0, 0]), scale, tuple([0, 0]))
            item = TrsF.affine(item, angle, tuple([0, 0]), 1, tuple([0, 0]))
            tdatalist.append(item)
        return tdatalist

class CropFuc(torch.nn.Module):
    def __init__(self,  size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.name = 'crop'
        self.paras = {'size': size, 'padding': padding, 'pad_if_needed': pad_if_needed, 'fill': fill, 'padding_mode': padding_mode}
        self.tempparams = {}
        self.funcClass = transforms.RandomCrop(**self.paras)

    def forward(self, itemlist, *args):
        Paras = self.funcClass.get_params(itemlist[0], self.paras['size'])
        self.tempparams.update(
            {
                'i': Paras[0],
                'j': Paras[1],
                'h': Paras[2],
                'w': Paras[3]
            }
        )
        tdatalist = []
        for item in itemlist:
            self.tempparams.update( { 'size': getSize(item) } )
            tdatalist.append(TrsF.crop(item, *Paras))
        return tdatalist
    
    # torch padding  [left, right, top, bottom]
    def invforward(self, itemlist, *args):
        i, j, h, w = self.tempparams['i'], self.tempparams['j'], self.tempparams['h'], self.tempparams['w']
        size = self.tempparams['size']
        top_ind, bottom_ind = i,  i + h
        left_ind, right_ind = j, j + w
        if isinstance(itemlist[0], torch.Tensor):
            left_pad, right_pad = left_ind, size[1] - right_ind
            top_pad, bottom_pad = top_ind, size[0] - bottom_ind
        tdatalist = []
        for item in itemlist:
            item = TrsF.resize(item, [h, w])
            if isinstance(item, torch.Tensor):
                tdatalist.append( F.pad(item, [left_pad, right_pad, top_pad, bottom_pad]) )
            else:
                tdata = np.zeros(size)
                tdata[top_ind: bottom_ind, left_ind: right_ind] = np.array(item)
                tdatalist.append( Image.fromarray(tdata) )
        return tdatalist

class MaskCropFuc(torch.nn.Module):
    def __init__(self,  size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.name = 'maskCrop'
        self.paras = { 'size': size, 'padding': padding, 'pad_if_needed': pad_if_needed, 'fill': fill, 'padding_mode': padding_mode }
        self.tempparams = { }
        self.funcClass = transforms.RandomCrop( **self.paras )
        self.resizeFunc = ResizeFuc( size )

    def forward(self, itemlist, mask, *args):
        if isinstance(mask, torch.Tensor):
            tmask = mask[0, 0].detach().cpu().numpy()
        else:
            tmask = np.array(tmask)
            
        ny, nx = np.nonzero(tmask)
        minx, maxx = min(nx), max(nx)
        miny, maxy = min(ny), max(ny)
        h, w = maxy - miny + 20, maxx - minx + 20
        oh, ow = self.paras['size']
        sqlen = np.max( [h, w, oh, ow] )
        
        x_c = minx - (sqlen - w) // 2
        y_c = miny - (sqlen - h) // 2
        if x_c < 0:
            x_c = 0
        if y_c < 0:
            y_c = 0
                
        self.tempparams.update(
            {
                'i': y_c,
                'j': x_c,
                'h': oh,
                'w': ow
            }
        )
        tdatalist = []
        for item in itemlist:
            self.tempparams.update( { 'size': getSize(item) } )
            item = TrsF.crop(item, y_c, x_c, oh, ow)
            tdatalist.append( item )
        tdatalist = self.resizeFunc( tdatalist )
        return tdatalist
    
    # torch padding  [left, right, top, bottom]
    def invforward(self, itemlist, *args):
        itemlist = self.resizeFunc.invforward( itemlist )
        i, j, h, w = self.tempparams['i'], self.tempparams['j'], self.tempparams['h'], self.tempparams['w']
        size = self.tempparams['size']
        top_ind, bottom_ind = i,  i + h
        left_ind, right_ind = j, j + w
        if isinstance(itemlist[0], torch.Tensor):
            left_pad, right_pad = left_ind, size[1] - right_ind
            top_pad, bottom_pad = top_ind, size[0] - bottom_ind
        tdatalist = []
        for item in itemlist:
            item = TrsF.resize(item, [h, w])
            if isinstance(item, torch.Tensor):
                tdatalist.append( F.pad(item, [left_pad, right_pad, top_pad, bottom_pad]) )
            else:
                tdata = np.zeros(size)
                tdata[top_ind: bottom_ind, left_ind: right_ind] = np.array(item)
                tdatalist.append( Image.fromarray(tdata) )
        return tdatalist

class HorizontalFlipFuc(torch.nn.Module):
    def __init__(self,  p = 0.5):
        super().__init__()
        self.name = 'hFlip'
        self.paras = {'p': p}
        self.tempparams = {}

    def forward(self, itemlist, *args):
        if self.paras['p'] > torch.rand(1):
            self.tempparams.update({'isflip': True})
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.hflip(item))
            return tdatalist
        else:
            self.tempparams.update({'isflip': False})
            return itemlist
    
    def invforward(self, itemlist, *args):
        if self.tempparams['isflip'] == True:
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.hflip(item))
            return tdatalist
        else:
            return itemlist

class RotationFuc(torch.nn.Module):
    def __init__(self,  degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None):
        super().__init__()
        self.name = 'rotation'
        self.paras = {'degrees': degrees, 'interpolation': interpolation, 'expand': expand, 'center': center, 'fill': fill, 'resample': resample}
        self.fucClass = transforms.RandomRotation(**self.paras)
        self.tempparams = {}

    def forward(self, itemlist, *args):
        Paras = self.fucClass.get_params(self.paras['degrees'])
        self.tempparas.update(
            {'angle': Paras}
            )
        tdatalist = []
        for item in itemlist:
            tdatalist.append(TrsF.rotate(item, Paras, self.paras['interpolation'], 
                                         self.paras['expand'], self.paras['center'], self.paras['fill'], self.paras['resample']))
        return tdatalist
    
    def invforward(self, itemlist, *args):
        angle = self.tempparas['angle'] * -1
        tdatalist = []
        for item in itemlist:
            tdatalist.append(TrsF.rotate(item, angle, self.paras['interpolation'], 
                                         self.paras['expand'], self.paras['center'], self.paras['fill'], self.paras['resample']))
        return tdatalist

class VerticalFlipFuc(torch.nn.Module):
    def __init__(self,  p = 0.5):
        super().__init__()
        self.name = 'vflip'
        self.paras = {'p': p}
        self.tempparams = {}

    def forward(self, itemlist, *args):
        if self.paras['p'] > torch.rand(1):
            self.tempparams.update({'isflip': True})
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.vflip(item))
            return tdatalist
        else:
            self.tempparams.update({'isflip': False})
            return itemlist
    
    def invforward(self, itemlist, *args):
        if self.tempparams['isflip'] == True:
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.vflip(item))
            return tdatalist
        else:
            return itemlist

class SlipWindowFuc(torch.nn.Module):
    def __init__(self,  imShape = [256, 256], winShape = [128, 128], stride = [1, 1]):
        super().__init__()
        self.name = 'slipWindows'
        self.imShape = imShape
        self.winShape = winShape
        self.tempparams = {}
        self.stride = stride
        tx, ty = 0, 0
        wh, ww = winShape
        ih, iw = imShape
        self.paraList = []
        while ty + wh < ih:
            tx = 0
            while tx + ww < iw:
                self.paraList.append([ty, tx, wh, ww])
                tx = tx + stride[1]
            tx = tx - (tx + ww - iw)
            self.paraList.append([ty, tx, wh, ww])
            ty = ty + stride[0]
        ty = ty - (ty + wh - ih)
        tx = 0
        while tx + ww < iw:
            self.paraList.append([ty, tx, wh, ww])
            tx = tx + stride[1]
        tx = tx - (tx + ww - iw)
        self.paraList.append([ty, tx, wh, ww])            

    def forward(self, itemlist, *args):
        tdatalist = []
        for item in itemlist:
            itemPatches = []
            for paras in self.paraList:
                itemPatches.append( TrsF.crop(item, *paras) )
            tdatalist.append(itemPatches)
        return tdatalist
    
    def invforward(self, itemlist, *args):
        ih, iw = self.imShape
        tdatalist = []
        for item in itemlist:
            patches = []
            for pi, patch in enumerate(item):                
                py, px, ph, pw = self.paraList[pi]
                top_ind, bottom_ind = py,  py + ph
                left_ind, right_ind = px, px + pw
                if isinstance(patch, torch.Tensor):
                    left_pad, right_pad = left_ind, iw - right_ind
                    top_pad, bottom_pad = top_ind, ih - bottom_ind
                    patches.append( F.pad( patch, [left_pad, right_pad, top_pad, bottom_pad] ) )
                else:
                    tdata = np.zeros( self.imShape )
                    tdata[top_ind: bottom_ind, left_ind: right_ind] = np.array( item )
                    patches.append( Image.fromarray(tdata) )
            tdatalist.append(patches)
        return tdatalist

def TTAAugList(
    im_shape, 
    patch_shape = None,
    patch_stride = None,
    degrees = [-15, 15],
    translate= [0.2, 0.2],
    scale=[1, 1], 
    shear=None,
    **kwargs
    ):
    return [
        ResizeCropFuc([im_shape, im_shape]),
        CropFuc([im_shape, im_shape]),
        HorizontalFlipFuc(1),
        VerticalFlipFuc(1),
        AffineFunc(
            degrees = degrees, 
            translate = translate, 
            scale = scale, 
            shear = shear, 
            interpolation = InterpolationMode.NEAREST, 
            fill = 0,
            fillcolor = None, 
            resample = None
            ),
        SlipWindowFuc([im_shape, im_shape], [256, 256], [128, 128])
        ]

import pickle as pkl
def debug():    
    datapath = 'E:/datasets/people/cornal/DPo/2-lidaizhen-13-91316-1_23.pkl'
    maskpath = 'E:/datasets/people/cornal/DM/2-lidaizhen-13-91316-1_23.pkl'
    datafile = open(datapath, 'rb')
    data = pkl.load(datafile)
    maskfile = open(maskpath, 'rb')
    mask = pkl.load(maskfile)
    img_ori = torch.tensor(data)
    img_ori = img_ori.unsqueeze(0).unsqueeze(0)
    mask_ori = torch.tensor(mask)
    mask_ori = mask_ori.unsqueeze(0).unsqueeze(0)
    translist = [     
        ResizeFuc([256, 256], InterpolationMode.NEAREST),
        ResizeCropFuc([256, 256]),
        MaskCropFuc([64, 64]),
        CropFuc([128, 128]),
        RotationFuc([-15, 15]),
        HorizontalFlipFuc(0.5),
        VerticalFlipFuc(0.5),                                  
        AffineFunc(degrees = [-15, 15], translate= [0.2, 0.2], scale=[0.8, 1.2], shear=[-15, 15, -15, 15], interpolation=InterpolationMode.NEAREST, fill=0,
            fillcolor=None, resample=None)
        ]
    transName = [
        'resize',
        'resizecrop',
        'maskcrop',
        'crop',
        'rotation',
        'hflip',
        'vfilp',
        'affine'
    ]
    guide = torch.zeros_like(img_ori) + 1
    resizefuc = ResizeFuc([384, 384], InterpolationMode.NEAREST)
    slipFuc = SlipWindowFuc([384, 384], [256, 256], [128, 128])
    img, mask, guide = resizefuc([img_ori, mask_ori, guide])
    img_patches, mask_patches, guide_patches = slipFuc([img, mask, guide])
    invimg, invmask, invguide = slipFuc.invforward([img_patches, mask_patches, guide_patches])
    
    Utils.SaveImageGrayOffset(
            [
                img.detach().cpu().numpy(),
                torch.cat(img_patches, 1).detach().cpu().numpy(),
                torch.cat(invimg, 1).sum(1, True).detach().cpu().numpy() / (torch.cat(invguide, 1).sum(1, True).detach().cpu().numpy() + 1e-5)
            ], 
            [
                'img',
                'img_patches',
                'invimg',
            ], 
            'C:/Users/wangl/Desktop/temp/aug_debug', 
            col_offset = 5,
            row_offset = 5,
            )
    
    Utils.SaveImageGrayOffset(
            [
                mask.detach().cpu().numpy(),
                torch.cat(mask_patches, 1).detach().cpu().numpy(),
                torch.cat(invmask, 1).sum(1, True).detach().cpu().numpy()/ (torch.cat(invguide, 1).sum(1, True).detach().cpu().numpy() + 1e-5)
            ], 
            [
                'mask',
                'mask_patches',
                'invmask',
            ], 
            'C:/Users/wangl/Desktop/temp/aug_debug', 
            col_offset = 5,
            row_offset = 5,
            )
    
    
    # for i in range(50):
    #     for trans, name in zip(translist, transName):
    #         print(i, name, img_ori.size())
            
    #         img, mask = trans([img_ori, mask_ori], mask_ori)
    #         invimg, invmask = trans.invforward([img, mask], mask_ori)
    #         [show_img, img, invimg, show_mask, mask, invmask] = reiszefuc([img_ori, img, invimg, mask_ori, mask, invmask])
    #         show_im = torch.cat([show_img, invimg, img], 1)
    #         show_mask = torch.cat([show_mask, invmask, mask], 1)
    #         show_data = torch.cat( [show_im, show_mask], 0 )
            
    #         Utils.SaveImageGrayOffset(
    #         [show_data.detach().cpu().numpy()
    #         ], 
    #         ['img_' + str(i) + '_' + name
    #         ], 
    #         'C:/Users/wangl/Desktop/temp/aug_debug', 
    #         col_offset = 5,
    #         row_offset = 5,
    #         col_num = show_data.size(0),
    #         row_num = show_data.size(1)
    #         )
        
if __name__ == '__main__':
    debug()