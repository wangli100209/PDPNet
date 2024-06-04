from config import Demo as config
from model import PDPNet
import torch
from utils import TrainerTester as TrainerTester
from utils import dataLoader
from utils import Saver as Saver
import os
import csv
from torch.nn import init

def Make_Dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def InitWeights(model, init_type='normal', gain=0.02):
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
                pass
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

def InitNet(model, init_type='normal', init_gain=0.02):
    InitWeights(model, init_type, gain=init_gain)

def LoadModel(path, model, device):
    print('loading ' + path) 
    model.load_state_dict( torch.load(path, map_location = device) )

def main():
    traning_set = dataLoader.dataSet(config.TR_ROOT)
    trainingloader = dataLoader.dataLoader(traning_set, 
                                           BatchSize = config.DEFAULT_BACTCHSIZE, 
                                           ShuffleFlag = 1, 
                                           ZeroOneFlag = True, 
                                           Patchsize = config.DEFAULT_PATCHSIZE, 
                                           device = config.DEFAULT_DEVICE, 
                                           data_aug = True)
    validing_set = dataLoader.dataSet(config.TE_ROOT)
    # PEOPLE_ROOT
    validingloader = dataLoader.dataLoader(validing_set, 
                                           BatchSize = config.DEFAULT_BACTCHSIZE, 
                                           ShuffleFlag = 1, 
                                           ZeroOneFlag = True, 
                                           Patchsize = config.DEFAULT_PATCHSIZE, 
                                           device = config.DEFAULT_DEVICE,
										   data_aug = False,
                                           droplast=False
										   )
    print('training loader len = ', len(trainingloader), ' validing loader len = ', len(validingloader))

    epoch = config.DEFAULT_EPOCH
    learning_rate = config.DEFAULT_LEARNINGRATE
    
    model = PDPNet.PDPNet()
    device = torch.device(config.DEFAULT_DEVICE)
    model = model.to(device)
    print('model biuld')

    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    trainer = TrainerTester.trainer(model, trainingloader, optim)
    valider = TrainerTester.valider(model, validingloader)
    saver = Saver.saver(config.SAVEER_PATH, 
                        save_rule = config.DEFAULT_SAVERULE, 
                        eraly_stop_times = config.DEFAULT_EARLYSTOPTIMES, 
                        cycletimes = config.DEFAULT_CYCLETIMES, 
                        metrics_function = TrainerTester.metrics_function)
    print('trainer valider saver initialized')
    
    if config.WARMUP_FLAG == True:
        saver.warm_up(config.CHECKPOINT_PATH, model, config.WUMETRIC_PATH, config.DEFAULT_DEVICE)
    else:
        InitNet(model, config.INIT_METHOD, config.INIT_GAIN)
        Make_Dir(os.path.split(config.TRAINLOGGFILE_PATH)[0])
        Make_Dir(os.path.split(config.TESTLOGGFILE_PATH)[0])
        tr_csvfile = open(config.TRAINLOGGFILE_PATH, 'w+', newline = '')
        tr_writer = csv.writer(tr_csvfile)
        te_csvfile = open(config.TESTLOGGFILE_PATH, 'w+', newline = '')
        te_writer = csv.writer(te_csvfile)
        keylist = ['epoch']
        for key, _ in saver.best_metrics.items():
            keylist.append(key)
        tr_writer.writerow(keylist)
        te_writer.writerow(keylist)
        tr_csvfile.close()
        te_csvfile.close()
    print('parameters initialized')

    print('training start')
    for e in range(epoch):
        tr_loss, tr_metrics = trainer.train_one_epoch(visual_step = config.DEFAULT_VISUALSTEP)
        value_list = [e]
        for _, value in tr_metrics.items():
            value_list.append(value)
        tr_csvfile = open(config.TRAINLOGGFILE_PATH, 'a+', newline = '')
        tr_writer = csv.writer(tr_csvfile)
        tr_writer.writerow(value_list)
        tr_csvfile.close()
        print('----------+----------+----------+----------+----------+----------')
        va_loss, va_metrics = valider.valid_one_epoch()
        value_list = [e]
        te_csvfile = open(config.TESTLOGGFILE_PATH, 'a+', newline = '')
        te_writer = csv.writer(te_csvfile)
        for _, value in va_metrics.items():
            value_list.append(value)
        te_writer.writerow(value_list)
        te_csvfile.close()
        print('----------*----------*----------*----------*----------*----------')
        saver.update(model, va_metrics)

if __name__ == '__main__':
    main()