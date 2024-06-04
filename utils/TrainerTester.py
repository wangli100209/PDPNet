import sys
sys.path.append('./')
sys.path.append('../')
from torch.nn import functional as F
import sys
import numpy as np
from utils import DictUtils, SegMetricUtils

def metrics_function(pre, mask):
    mask_thres = F.adaptive_avg_pool2d(mask.float(), [1, 1])
    label = (mask > mask_thres).long() * (mask != 1).long() + (mask == 1).long()
    # label = (mask > 0).long()
    # select_metrics = [
    #     'DSC', 'JAC', 'SEN', 'SPE', 'FPR', 'FNR',
    #     'PPV', 'F1', 'GCE', 'VD', 'VS', 'MI', 'VOI',
    #     'ICC', 'PBD', 'KAPPA', 'HD', 'HD95', 'AVD',
    #     'ASD', 'ASSD'
    # ]
    select_metrics = [
        'DSC', 'SEN', 'SPE'
    ]
    seg_m = SegMetricUtils.seg_metrics(
        pre,
        label,
        start_index = 1,
        ForSaver = False,
        select_metrics = select_metrics
        )
    return seg_m

class trainer():
    def __init__(self, model, loader, optims):
        super(trainer,self).__init__()
        self.model = model
        self.loader = loader
        self.optims = optims
        self.metrics_function = metrics_function
        self.dict_util = DictUtils.DictUtil()
        self.epoch_now = 0
    
    def train_one_step(self, data, mask):
        logits, crop_datas, losses = self.model(data, mask)
        
        loss_total = 0
        for loss in losses:
            loss_total = loss_total + loss
        self.optims.zero_grad()
        loss_total.backward()
        self.optims.step()

        metrics = self.metrics_function(logits[-1], mask)
        lossitemlist = []
        for i, loss in enumerate(losses):
            lossitemlist.append(loss.data.item())
        return lossitemlist, metrics

    def train_one_epoch(self, visual_step = 100):        
        self.model.train()
        total_loss = []
        total_metrics = []
        for i, batch_datas in enumerate(self.loader):
            postdata, mask, _ = batch_datas
            step_loss, step_metrics = self.train_one_step(postdata, mask)
            total_loss.append(np.mean(step_loss))
            total_metrics.extend(step_metrics)
            if i % visual_step == 0 and i != 0:
                print('training step:  ', i, '  avg_loss: ', np.mean(total_loss))
                if self.metrics_function != None:
                    self.dict_util.PrintDict(self.dict_util.MeanDictList(total_metrics), 5)
        self.epoch_now += 1
        re_loss = np.mean(total_loss)
        re_metrics= self.dict_util.MeanDictList(total_metrics)
        print('training epoch:  ', self.epoch_now, '  avg_loss: ', re_loss)
        if self.metrics_function != None:
            self.dict_util.PrintDict(re_metrics, 5)
        return re_loss, re_metrics

class valider():
    def __init__(self, model, loader):
        super(valider,self).__init__()
        self.loader = loader
        self.model = model
        self.metrics_function = metrics_function
        self.dict_util = DictUtils.DictUtil()
        self.epoch_now = 0

    def valid_one_step(self, data, mask):
        logits, crop_datas, losses = self.model(data, mask)
        metrics = self.metrics_function(logits[-1], mask)
        lossitemlist = []
        for i, loss in enumerate(losses):
            lossitemlist.append(loss.data.item())
        return lossitemlist, metrics

    def valid_one_epoch(self):
        self.model.eval()
        total_loss = []
        total_metrics = []
        for i, batch_datas in enumerate(self.loader):
            postdata, mask, _ = batch_datas
            step_loss, step_metrics = self.valid_one_step(postdata, mask)
            total_loss.append(np.mean(step_loss))
            total_metrics.extend(step_metrics)
        self.epoch_now += 1
        re_loss = np.mean(total_loss)
        re_metrics = self.dict_util.MeanDictList(total_metrics)
        print('validing epoch:  ', self.epoch_now, '  avg_loss: ', re_loss)
        if self.metrics_function != None:
            self.dict_util.PrintDict(re_metrics, 5)
        return re_loss, re_metrics
