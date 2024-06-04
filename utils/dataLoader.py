import os
import numpy as np
import torch
import math
from utils import myTransform
from torchvision import transforms
from PIL import Image
import pickle

def load_pickle(path):
    f = open(path, 'rb')
    file = pickle.load(f)
    return file

def Zeroscore_Normalization(data):
    return (data - np.mean(data)) / np.std(data)

def ZeroOne_Normalization(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-5)

def SYNPartition_Center(maindata, subdatalist, center, patchsize):
    Ddim = len(np.shape(maindata))
    if Ddim == 2:
        h, w = np.shape(maindata)
        
        x = center[0]
        y = center[1]

        ph, pw = patchsize
        ph = np.minimum(ph, h)
        pw = np.minimum(pw, w)
        
        if x + ph // 2 > h:
            x = x - (x + ph // 2 - h)
        elif x - ph // 2 < 0:
            x = x + (ph // 2 - x)
    
        if y + pw // 2 > w:
            y = y - (y + pw // 2 - w)
        elif y - pw // 2 < 0:
            y = y + (pw // 2 - y)
            
        tmaindata = maindata[x - ph // 2 : x + ph // 2, y - pw // 2 : y + pw // 2]
        tsubdatalist = []
        for subdata in subdatalist:
            tsubdatalist.append(subdata[x - ph // 2 : x + ph // 2, y - pw // 2 : y + pw // 2])
        return tmaindata, tsubdatalist
    
    if Ddim == 3:
        h, w, d = np.shape(maindata)
        
        x = center[0]
        y = center[1]
        z = center[2]

        ph, pw, pd = patchsize
        ph = np.minimum(ph, h)
        pw = np.minimum(pw, w)
        pd = np.minimum(pd, d)
        
        if x + ph // 2 > h:
            x = x - (x + ph // 2 - h)
        elif x - ph // 2 < 0:
            x = x + (ph // 2 - x)
    
        if y + pw // 2 > w:
            y = y - (y + pw // 2 - w)
        elif y - pw // 2 < 0:
            y = y + (pw // 2 - y)
            
        if z + pd // 2 > d:
            z = z - (z + pd // 2 - z)
        elif z - pd // 2 < 0:
            z = z + (pd // 2 - z)
        tmaindata = maindata[x - ph // 2 : x + ph // 2, y - pw // 2 : y + pw // 2, z - pd // 2 : z + pd // 2]
        tsubdatalist = []
        for subdata in subdatalist:
            tsubdatalist.append(subdata[x - ph // 2 : x + ph // 2, y - pw // 2 : y + pw // 2, z - pd // 2 : z + pd // 2])
        return tmaindata, tsubdatalist

def trdata_aug():
    return transforms.Compose([                                                  
                myTransform.AdjustSharpnessFunc(1, 0.5),
                # myTransform.GaussianBlurFuc(3, (0.1, 2.0)),
                myTransform.NoiseFuc('gaussian'),
                ])

def trall_aug():
    return transforms.Compose([           
                myTransform.HorizontalFlipFuc(0.5),
                # myTransform.VerticalFlipFuc(0.5),                                    
                myTransform.AffineFunc(degrees = [-15, 15], translate= [0.2, 0.2], scale=[1, 1], shear=None, interpolation=transforms.InterpolationMode.BILINEAR, fill=0,
                    fillcolor=None, resample=None),
                # ResizeCropFuc((128, 128)),
                ])

def evalall_aug():
    return transforms.Compose([           
                # myTransform.HorizontalFlipFuc(0.5),
                myTransform.VerticalFlipFuc(1),                                    
                # myTransform.AffineFunc(degrees = [-15, 15], translate= [0.2, 0.2], scale=[1, 1], shear=None, interpolation=transforms.InterpolationMode.BICUBIC, fill=0,
                #     fillcolor=None, resample=None),
                # ResizeCropFuc((128, 128)),
                ])

def loader(dataroot, file, index):
    item = load_pickle(os.path.join(dataroot, file))
    data = item['Data']
    mask = item['GT']
    if index == -1:
        if len(data.shape) == 3:
            data = np.mean(data, -1)
        if len(mask.shape) == 3:
            mask = np.mean(mask, -1)
    else:
        if len(data.shape) == 3:
            if index >= data.shape[2]:
                index = -1
            data = data[..., index]
        if len(mask.shape) == 3:
            if index >= mask.shape[2]:
                index = -1
            mask = mask[..., index]
    return data, mask
        
def SYNShuffle(mainlist, sublists):
    state = np.random.get_state()
    np.random.shuffle(mainlist)
    for sublist in sublists:
        np.random.set_state(state)
        np.random.shuffle(sublist)
    
def to_tensor(data, device):
    return torch.tensor(data, dtype = torch.float32, device = device)

def MyOneHot(mask, num_classes):
    h, w = np.shape(mask)
    one_hot = np.zeros([num_classes, h, w])
    for c in range(num_classes):
        x, y = np.where(mask  == c)
        one_hot[c, x, y] = 1
    return one_hot

## setFlag = 0 all data  1 training data   2 testing data
class dataSet():
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.datalist = np.array( os.listdir(dataroot) )
        
        self.loader = loader
        self.shuffleFuc = SYNShuffle
        self.zerooneFuc = ZeroOne_Normalization
        self.patchFuc = SYNPartition_Center
        self.to_tensorFuc = to_tensor
        self.onehotFuc = MyOneHot
    
    def __len__(self):
        return len(self.datalist)

#ShuffleFlag  0 no shuffle   1 shuffle one time   2 shuffle every time
class dataLoader():
    def __init__(self, dataset, BatchSize = 1, ShuffleFlag = 0, ZeroOneFlag = False, Patchsize = None, device = 'cuda', droplast = True, data_aug = 1, loaderind = -1):
        self.dataset = dataset
        self.samplingList = np.arange(len(self.dataset))
        self.indexnum = 0
        self.batchsize = BatchSize
        self.ShuffleFlag = ShuffleFlag
        if self.ShuffleFlag == 1:
            self.dataset.shuffleFuc(self.dataset.datalist, [])
        self.ZeroOneFlag = ZeroOneFlag
        self.Patchsize = Patchsize
        self.device = device
        self.droplast = droplast
        self.maskthres = 0.005
        # self.maskthres = 0
        self.data_aug = data_aug
        self.trdata_aug = trdata_aug()
        self.trall_aug = trall_aug()
        self.eval_aug = evalall_aug()
        self.resizefuc = myTransform.ResizeFuc((128, 128))
        self.loaderind = loaderind
    
    def __len__(self):
        if self.droplast:
            return math.ceil(len(self.dataset) / self.batchsize) - 1
        else:
            return math.ceil(len(self.dataset) / self.batchsize)
    
    def __iter__(self):
        self.indexnum = 0
        return self

    def __next__(self):
        if self.indexnum >= len(self.dataset):
            if self.ShuffleFlag == 2:
                self.dataset.shuffleFuc(self.dataset.datalist, [])
            raise StopIteration
        elif self.indexnum + self.batchsize >= len(self.dataset) and self.droplast == True:
            if self.ShuffleFlag == 2:
                self.dataset.shuffleFuc(self.dataset.datalist, [])
            raise StopIteration
        else:
            data = []
            mask = []
            for i in range(self.batchsize):
                if self.indexnum < len(self.dataset):
                    tindex = self.samplingList[self.indexnum]
                    datas = self.dataset.loader(self.dataset.dataroot, self.dataset.datalist[tindex], self.loaderind)
                    if datas != None:
                        tdata, tmask = datas
                        if np.sum(tmask) / np.size(tmask) <= self.maskthres:
                            tmask = np.zeros_like(tmask)
                        tdata = self.dataset.zerooneFuc(tdata)     
                        tdata = Image.fromarray( (tdata * 255).astype(np.uint8) )
                        tmask = Image.fromarray(tmask)
                        tdata, tmask = self.resizefuc([tdata, tmask])
                        if self.data_aug == 1:
                            tdata, tmask = self.trall_aug([tdata, tmask])
                        elif self.data_aug == 2:
                            tdata, tmask = self.eval_aug([tdata, tmask])
                        else:
                            tdata, tmask = tdata, tmask
                        if self.ZeroOneFlag == True:
                            tdata = np.array(tdata).astype(np.float32) / 255
                        else:
                            tdata = np.array(tdata).astype(np.float32)
                        tmask = np.array(tmask)
                        # tmask = (tmask > np.mean(tmask)).astype(np.int16)
                        if self.Patchsize != None:
                            x, y = np.nonzero(tmask)
                            if len(x) != 0:
                                minx, maxx = np.min(x), np.max(x)
                                miny, maxy = np.min(y), np.max(y)
                                tdata, subdata = self.dataset.patchFuc(tdata, [tmask], [(maxx - minx) // 2, (maxy - miny) // 2], [self.Patchsize[0], self.Patchsize[1]])
                                #tdata, subdata = self.dataset.patchFuc(tdata, [tmask], [self.Patchsize[0], self.Patchsize[1]])
                                tmask = subdata[0]
                                # tpredata = subdata[1]
                            else:
                                # tdata, subdata = self.dataset.patchFuc(tdata, [tmask, tpredata], [np.shape(tdata)[0] // 2, np.shape(tdata)[1] // 2], [self.Patchsize[0], self.Patchsize[1]])
                                tdata, subdata = self.dataset.patchFuc(tdata, [tmask], [128, 128], [self.Patchsize[0], self.Patchsize[1]])
                                tmask = subdata[0]
                                # tpredata = subdata[1]
                        self.indexnum += 1
                        data.append(np.array(tdata)[np.newaxis, np.newaxis, ...])
                        # tmask = MyOneHot(tmask, 2)
                        mask.append(np.array(tmask)[np.newaxis, np.newaxis, ...])
                        # predata.append(np.array(tpredata)[np.newaxis, np.newaxis, ...])
                    else:
                        self.indexnum += 1
                else:
                    return self.dataset.to_tensorFuc(np.concatenate(data, 0), self.device), \
								self.dataset.to_tensorFuc(np.concatenate(mask, 0), self.device).long(), \
                                None
        
            return self.dataset.to_tensorFuc(np.concatenate(data, 0), self.device), \
                        self.dataset.to_tensorFuc(np.concatenate(mask, 0), self.device).long(), \
                        None

def main():
    # dataset = dataSet('E:/datasets/SliceData/HUM_231101/3')
    dataset = dataSet('F:/datasets/SlicData/HUM/items2d')
    dataloader = dataLoader(dataset, BatchSize = 32, ShuffleFlag = 1, ZeroOneFlag = True, Patchsize = None, device = 'cpu', data_aug=False)
    print(len(dataloader))
    for i in range(1):
        for j, v in enumerate(dataloader):
            print( i, j, np.shape(v[0]), np.shape(v[1]) )
            # Utils.SaveImageGray(
            #     [
            #         v[0].detach().cpu().numpy(),
            #         v[1].detach().cpu().numpy(),
            #         (v[0] * v[1]).detach().cpu().numpy()
            #     ],
            #     [
            #         'data_' + str(j),
            #         'mask_' + str(j),
            #         'roi_' + str(j)
            #     ], 
            #     './debug'
            #     )

if __name__ == "__main__":
    main()
