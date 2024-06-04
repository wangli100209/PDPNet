import sys
sys.path.append('./')
sys.path.append('../')
from utils import DictUtils
import os
import torch
import pickle


def SaveModel(path, model):
    torch.save(model.state_dict(), path)

def LoadModel(path, model, device):
    print('loading ' + path) 
    model.load_state_dict( torch.load(path, map_location = device) )

def save_pickle(path, file):
    f = open(path, 'wb')
    pickle.dump(file, f)
    f.close()
    
def load_pickle(path):
    f = open(path, 'rb')
    file = pickle.load(f)
    return file

def Make_Dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def FName_Normalization(name, namelens = 6):
    zerolens = namelens - len(name)
    filename = ''
    for i in range(zerolens):
        filename += '0'
    filename += name
    return filename
# save_rule:  cycle  rewrite by cycletimes
#                   metric  rewrite by best metrcis
#                   cycle_metric  rewrite by both cycletimes and metrics
class saver():
    def __init__(self, save_root, save_rule = 'cycle', eraly_stop_times = 50, cycletimes = 3, metrics_function = None, metrics_w = None, metrics_com_rule = None):
        super(saver,self).__init__()
        self.save_root = save_root
        self.save_rule = save_rule
        self.metrics_function = metrics_function
        self.dict_util = DictUtils.DictUtil()
        self.metrics_w = metrics_w
        self.metrics_com_rule = metrics_com_rule
        self.cycletimes = cycletimes
        self.cyclenow = 0
        self.eraly_stop_times = eraly_stop_times
        self.early_stop_now = 0
        self.best_metrics = self.init_metrics()
        Make_Dir(self.save_root)

    def init_metrics(self):
        if self.metrics_function == None:
            return {'loss': -1000}
        else:
            pre = torch.zeros(1, 1, 8, 8)
            mask = torch.ones(1, 1, 8, 8)
            return self.dict_util.MeanDictList( self.metrics_function(pre, mask) )

    def update(self, model, metrics):
        if self.save_rule == 'cycle' or self.save_rule == 'cycle_metric':
            save_pickle(os.path.join(self.save_root, FName_Normalization(str(self.cyclenow)) + '_metrics.pkl'), metrics)
            SaveModel(os.path.join(self.save_root, FName_Normalization(str(self.cyclenow)) + '_model.pth'), model)
            if self.cyclenow == self.cycletimes:
                self.cyclenow = 0
            self.cyclenow += 1
        if self.save_rule == 'metric' or self.save_rule == 'cycle_metric':
            com = self.dict_util.CompareDicts(metrics, self.best_metrics)
            if com >= 0.5:
                self.best_metrics.update(metrics)
                self.early_stop_now = 0
                save_pickle(os.path.join(self.save_root, 'best_metrics.pkl'), metrics)
                SaveModel(os.path.join(self.save_root, 'best_model.pth'), model)
            else:
                self.early_stop_now += 1
                if self.early_stop_now == self.eraly_stop_times:
                    exit(0)

    def warm_up(self, checkpoint_path, model, metrics_path):
        LoadModel(checkpoint_path, model)
        self.best_metrics.update(load_pickle(metrics_path))