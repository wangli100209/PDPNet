import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TrsF
from torchvision.transforms import InterpolationMode
import torch
from skimage.util import random_noise

class ResizeFuc(torch.nn.Module):
    def __init__(self,  size, interpolation = 2):
        super().__init__()
        self.paras = {'size': size, 'interpolation': interpolation}

    def forward(self, itemlist):
        tdatalist = []
        for item in itemlist:
            tdatalist.append(TrsF.resize(item, **self.paras))
        return tdatalist
    
class ResizeCropFuc(torch.nn.Module):
    def __init__(self,  size, scale = (0.08, 1.0), ratio = (3. / 4., 4. / 3.), interpolation = InterpolationMode.BICUBIC):
        super().__init__()
        self.paras = {'size': size, 'scale': scale, 'ratio': ratio, 'interpolation': interpolation}
        self.funcClass = transforms.RandomResizedCrop(**self.paras)

    def forward(self, itemlist):
        Paras = self.funcClass.get_params(itemlist[0], self.funcClass.scale, self.funcClass.ratio)
        tdatalist = []
        for item in itemlist:
            item = TrsF.crop(item, *Paras)
            item = TrsF.resize(item, self.paras['size'], self.paras['interpolation'])
            tdatalist.append(item)
        return tdatalist

class AdjustSharpnessFunc(torch.nn.Module):
    def __init__(self,  sharpness_factor, p=0.5):
        super().__init__()
        self.paras = {'sharpness_factor': sharpness_factor, 'p': p}

    def forward(self, itemlist):
        if self.paras['p'] > torch.rand(1):
            tdatalist = []
            for item in itemlist:
                item=item.convert('L')
                item = TrsF.adjust_sharpness(item, self. paras['sharpness_factor'])
                tdatalist.append(item)
            return tdatalist
        else:
            return itemlist

class AffineFunc(torch.nn.Module):
    def __init__(self,  degrees, translate=None, scale=None, shear=None, interpolation=InterpolationMode.NEAREST, fill=0,
        fillcolor=None, resample=None):
        super().__init__()
        self.paras = {'degrees': degrees, 'translate': translate, 'scale': scale, 'shear' : shear, 'interpolation': InterpolationMode.BICUBIC,
                'fill': fill
             }
        self.funcClass = transforms.RandomAffine(**self.paras)

    def forward(self, itemlist):
        Paras = self.funcClass.get_params(self.funcClass.degrees, 
                                          self.funcClass.translate, 
                                          self.funcClass.scale, 
                                          self.funcClass.shear, itemlist[0].size)
        tdatalist = []
        for item in itemlist:
            item = TrsF.affine(item, *Paras)
            tdatalist.append(item)
        return tdatalist
    
class AutocontrastFuc(torch.nn.Module):
    def __init__(self,  p):
        super().__init__()
        self.paras = {'p': p}

    def forward(self, itemlist):
        if self.paras['p'] > torch.rand(1):
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.autocontrast(item.convert('L')))
            return tdatalist
        else:
            return itemlist

class CropFuc(torch.nn.Module):
    def __init__(self,  size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.paras = {'size': size, 'padding': padding, 'pad_if_needed': pad_if_needed, 'fill': fill, 'padding_mode': padding_mode}
        self.funcClass = transforms.RandomCrop(**self.paras)

    def forward(self, itemlist):
        Paras = self.funcClass.get_params(itemlist[0], self.paras['size'])
        tdatalist = []
        for item in itemlist:
            tdatalist.append(TrsF.crop(item, *Paras))
        return tdatalist

class EqualizeFuc(torch.nn.Module):
    def __init__(self,  p):
        super().__init__()
        self.paras = {'p': p}

    def forward(self, itemlist):
        if self.paras['p'] > torch.rand(1):
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.equalize(item.convert('L')))
            return tdatalist
        else:
            return itemlist

class EraseFuc(torch.nn.Module):
    def __init__(self,  p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()
        self.paras = {'p': p, 'scale': scale, 'ratio': ratio, 'value': value, 'inplace': inplace}
        self.fucClass = transforms.RandomErasing(**self.paras)

    def forward(self, itemlist):
        if self.paras['p'] > torch.rand(1):
            Paras = self.fucClass.get_params(torch.tensor(np.array(itemlist[0])[np.newaxis, ...]), 
                                             scale=self.paras['scale'], ratio=self.paras['ratio'], value=self.paras['value'])
            tdatalist = []
            for item in itemlist:
                item = torch.tensor(np.array(item)[np.newaxis, ...])
                print(item.shape)
                item = TrsF.erase(item, *Paras, self.paras['inplace'])
                tdatalist.append(Image.fromarray(item.detach().cpu().numpy()[0]))
            return tdatalist
        else:
            return itemlist

class HorizontalFlipFuc(torch.nn.Module):
    def __init__(self,  p = 0.5):
        super().__init__()
        self.paras = {'p': p}

    def forward(self, itemlist):
        if self.paras['p'] > torch.rand(1):
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.hflip(item))
            return tdatalist
        else:
            return itemlist

class InvertFuc(torch.nn.Module):
    def __init__(self,  p = 0.5):
        super().__init__()
        self.paras = {'p': p}

    def forward(self, itemlist):
        if self.paras['p'] > torch.rand(1):
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.invert(item.convert('L')))
            return tdatalist
        else:
            return itemlist

class PerspectiveFuc(torch.nn.Module):
    def __init__(self,  distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BICUBIC, fill=0):
        super().__init__()
        self.paras = {'distortion_scale': distortion_scale, 'p': p, 'interpolation': interpolation, 'fill': fill}
        self.fucClass = transforms.RandomPerspective(**self.paras)

    def forward(self, itemlist):
        if self.paras['p'] > torch.rand(1):
            Paras = self.fucClass.get_params(itemlist[0].size[0], itemlist[0].size[1], self.paras['distortion_scale'])
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.perspective(item, *Paras, self.paras['interpolation'], self.paras['fill']))
            return tdatalist
        else:
            return itemlist

class RotationFuc(torch.nn.Module):
    def __init__(self,  degrees, interpolation=InterpolationMode.BICUBIC, expand=False, center=None, fill=0, resample=None):
        super().__init__()
        self.paras = {'degrees': degrees, 'interpolation': interpolation, 'expand': expand, 'center': center, 'fill': fill, 'resample': resample}
        self.fucClass = transforms.RandomRotation(**self.paras)

    def forward(self, itemlist):
        Paras = self.fucClass.get_params(self.paras['degrees'])
        tdatalist = []
        for item in itemlist:
            tdatalist.append(TrsF.rotate(item, Paras, self.paras['interpolation'], 
                                         self.paras['expand'], self.paras['center'], self.paras['fill'], self.paras['resample']))
        return tdatalist

class SolarizeFuc(torch.nn.Module):
    def __init__(self,  threshold, p=0.5):
        super().__init__()
        self.paras = {'threshold': threshold, 'p': p}
        self.fucClass = transforms.RandomSolarize(**self.paras)

    def forward(self, itemlist):
        if self.paras['p'] > torch.rand(1):
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.solarize(item.convert('L'), self.paras['threshold']))
            return tdatalist
        else:
            return itemlist

class VerticalFlipFuc(torch.nn.Module):
    def __init__(self,  p = 0.5):
        super().__init__()
        self.paras = {'p': p}

    def forward(self, itemlist):
        if self.paras['p'] > torch.rand(1):
            tdatalist = []
            for item in itemlist:
                tdatalist.append(TrsF.vflip(item))
            return tdatalist
        else:
            return itemlist

class ColorJitterFuc(torch.nn.Module):
    def __init__(self,  brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.paras = {'brightness': brightness, 'contrast': contrast, 'saturation': saturation, 'hue': hue}
        self.fucClass = transforms.ColorJitter(**self.paras)

    def forward(self, itemlist):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor =  \
        self.fucClass.get_params(self.fucClass.brightness, self.fucClass.contrast, self.fucClass.saturation, self.fucClass.hue)
        tdatalist = []
        for item in itemlist:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    item = TrsF.adjust_brightness(item.convert('L'), brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    item = TrsF.adjust_contrast(item.convert('L'), contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    item = TrsF.adjust_saturation(item.convert('L'), saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    item = TrsF.adjust_hue(item.convert('L'), hue_factor)
            tdatalist.append(item)
        return tdatalist

class GaussianBlurFuc(torch.nn.Module):
    def __init__(self,  kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.paras = {'kernel_size': kernel_size, 'sigma': sigma}
        self.fucClass = transforms.GaussianBlur(**self.paras)

    def forward(self, itemlist):
        sigma = self.fucClass.get_params(*self.fucClass.sigma)
        tdatalist = []
        for item in itemlist:
            tdatalist.append(TrsF.gaussian_blur(item.convert('L'), self.paras['kernel_size'], sigma))
        return tdatalist

def gen_random_noise(shape, mode):
    noise = np.zeros(shape)
    noise = random_noise(noise, mode)
    return noise

class NoiseFuc(torch.nn.Module):
    def __init__(self,  mode):
        super().__init__()
        self.paras = {'mode': mode}

    def forward(self, itemlist):
        noise = gen_random_noise(itemlist[0].size, self.paras['mode'])
        tdatalist = []
        for item in itemlist:
            item = np.array(item)
            item = (item - np.min(item)) / (np.max(item) - np.min(item) + 1e-5)
            item = item + noise
            tdatalist.append(Image.fromarray(item))
        return tdatalist