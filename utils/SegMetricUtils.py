import sys
sys.path.append('./')
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from utils import MySegMetrics
import torch

def debug():
    ['DSC', 'JAC', 'SEN', 'SPE', 'FPR', 'FNR',
     'PPV', 'F1', 'GCE', 'VD', 'VS', 'MI', 'VOI',
     'ICC', 'PBD', 'KAPPA', 'HD', 'HD95', 'AVD',
     'ASD', 'ASSD'
     ]
    
def get_MetricsFunc():
    return {
        'DSC': MySegMetrics.DiceCoefficient, 
        'JAC': MySegMetrics.JaccardCoefficient, 
        'SEN': MySegMetrics.Recall, 
        'SPE': MySegMetrics.Specificity, 
        'FPR': MySegMetrics.FPR, 
        'FNR': MySegMetrics.FNR, 
        'PPV': MySegMetrics.Precision, 
        'F1': MySegMetrics.F1Score, 
        'GCE': MySegMetrics.GlobalConsistencyError, 
        'VD': MySegMetrics.VolumetricDistance, 
        'VS': MySegMetrics.VolumtricSimilarity, 
        'MI': MySegMetrics.MutualInformation, 
        'VOI': MySegMetrics.VariationofInformation, 
        'ICC': MySegMetrics.InterclassCorrelation, 
        'PBD': MySegMetrics.ProbabilisticDistance, 
        'KAPPA': MySegMetrics.Kappa, 
        'HD': MySegMetrics.HausdorffDistance, 
        'HD95': MySegMetrics.HausdorffDistance95, 
        'AVD': MySegMetrics.AverageHausdorffDistance,
        'ASD': MySegMetrics.AverageSurfaceDistance, 
        'ASSD': MySegMetrics.AverageSymmetricSurfaceDistance
    }

def get_DistanceMetricsKey():
    return ['MI', 'PBD', 'HD', 'VD', 'HD95', 'AVD', 'ASD', 'ASSD']

def seg_metrics(
    pre, 
    target, 
    start_index = 0, 
    ForSaver = True,
    select_metrics = [
    'DSC', 'JAC', 'SEN', 'SPE', 'FPR', 'FNR',
    'PPV', 'F1', 'GCE', 'VD', 'VS', 'MI', 'VOI',
    'ICC', 'PBD', 'KAPPA', 'HD', 'HD95', 'AVD',
    'ASD', 'ASSD'
    ]
    ):
    metricsFucs = get_MetricsFunc()
    distaceKeys = get_DistanceMetricsKey()
    
    if pre is None or target is None:
        non_Dict = {
            'DSC': 0, 'JAC': 0, 'SEN': 0, 'SPE': 0, 
            'FPR': 0, 'FNR': 0, 'PPV': 0, 'F1': 0, 
            'GCE': 0, 'VD': 0, 'VS': 0, 'MI': 65535, 
            'VOI': 0, 'ICC': 0, 'PBD': 65535, 'KAPPA': 0, 
            'HD': 65535, 'HD95': 65535, 'AVD': 65535,
            'ASD': 65535, 'ASSD': 65535
            }
        re_dict = {}
        for mname in select_metrics:
            re_dict[ mname ] = non_Dict[ mname ]
        return [re_dict]
    
    if pre.size(1) != 1:
        num_class = pre.size(1)
        # B, H, W, D
        tpre = torch.argmax(pre, dim = 1).detach().cpu().numpy()
        # B, H, W, D
        ttarget = target.squeeze(1).detach().cpu().numpy()
    else:
        num_class = 2
        tpre = (pre > 0.5).long()
        tpre = tpre.squeeze(1).detach().cpu().numpy()
        ttarget = target.squeeze(1).detach().cpu().numpy()
    sum_axis = tuple( np.arange( 1, len(ttarget.shape) ) )
    targetsum = np.sum(ttarget, sum_axis)
    presum = np.sum(tpre, sum_axis)
    
    tpre = MySegMetrics.one_hot(tpre, num_class)
    ttarget = MySegMetrics.one_hot(ttarget, num_class)
    if len(np.where(targetsum != 0)[0]) != 0:
        metrics_list = []
        for index in np.where(targetsum != 0)[0]:
            tmetrics_dict = {}
            if np.sum(presum[index]) == 0:
                dtpre = np.zeros_like(tpre[index])
                dtpre[pre.size(2) // 2, pre.size(3) // 2] = 1
            else:
                dtpre = tpre[index]
            for mname in select_metrics:
                if mname not in distaceKeys:
                    tmetrics = metricsFucs[mname](tpre[index], ttarget[index])
                    tmetrics_dict[mname] = np.mean(tmetrics[ start_index: ])
                else:
                    tmetrics = metricsFucs[mname](dtpre, ttarget[index])
                    if ForSaver == True:
                        tmetrics_dict[mname] = np.mean(tmetrics[ start_index: ]) * -1
                    else:
                        tmetrics_dict[mname] = np.mean(tmetrics[ start_index: ])
            metrics_list.append(
                tmetrics_dict
            )
        return metrics_list
    else:
        return None
    
    