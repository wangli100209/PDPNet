import sys
sys.path.append('./')
sys.path.append('../')
import numpy as np
import math
import medpy.metric.binary as bn
import torch
from sklearn.metrics.cluster._supervised import rand_score, adjusted_rand_score, mutual_info_score
from utils import SobelTorch

eps = 1e-5

def one_hot(pre, num_class):
    tpre = np.zeros_like(pre, dtype = np.uint8)[..., np.newaxis]
    tpre = np.repeat(tpre, num_class, -1)
    for i in range(num_class):
        tpre[..., i][np.where(pre == i)] = 1
    return tpre

def ConfusionMatrix(pre, target):    
    dim = len(target.shape)
    tpre = np.expand_dims( pre, -1 )
    ttarget = np.expand_dims( target, -2 )
    inter = ttarget * tpre
    sum_axis = tuple( np.arange(dim - 1) )
    cmatrix = np.sum(inter, sum_axis)
    return cmatrix

def get_TP(pre, target):
    cmatrix = ConfusionMatrix(pre, target)
    m_eye = cmatrix * np.eye( cmatrix.shape[0] )
    return np.sum(m_eye, 1)
    
def get_FP(pre, target):
    cmatrix = ConfusionMatrix(pre, target)
    m_eye = cmatrix * np.eye( cmatrix.shape[0] )
    return np.sum(cmatrix, 1) - np.sum(m_eye, 1)

def get_FN(pre, target):
    cmatrix = ConfusionMatrix(pre, target)
    m_eye = cmatrix * np.eye( cmatrix.shape[0] )
    return np.sum(cmatrix, 0) - np.sum(m_eye, 1)

def get_TN(pre, target):
    cmatrix = ConfusionMatrix(pre, target)
    m_eye = cmatrix * np.eye( cmatrix.shape[0] )
    tp = np.sum(m_eye, 1)
    fp = np.sum(cmatrix, 1) - np.sum(m_eye, 1)
    fn = np.sum(cmatrix, 0) - np.sum(m_eye, 1)
    return np.sum( cmatrix ) - tp - fp - fn

def DiceCoefficient(pre, target):
    dim = len(target.shape)
    sum_axis = tuple(np.arange(dim - 1))
    dice = 2 * np.sum(pre * target, sum_axis) / \
        ( np.sum(pre, sum_axis) + np.sum(target, sum_axis) + eps )
    return dice

def JaccardCoefficient(pre, target):
    dim = len(target.shape)
    inter = pre * target
    union = pre + target
    union[ np.where(union > 0) ] = 1
    sum_axis = tuple(np.arange(dim - 1))
    jc = np.sum(inter, sum_axis) / \
        ( np.sum(union, sum_axis) + eps )
    return jc

# Recall Sensitivity TPR
def Recall(pre, target):
    tp = get_TP(pre, target)
    fn = get_FN(pre, target)
    return tp / ( tp + fn + eps )

#Specificity TNR
def Specificity(pre, target):
    tn = get_TN(pre, target)
    fp = get_FP(pre, target)
    return tn / (tn + fp + eps)

#Fallout FPR
def FPR(pre, target):
    return 1 - Specificity(pre, target)

def FNR(pre, target):
    return 1 - Recall(pre, target)

#Precision PPV
def Precision(pre, target):
    tp = get_TP(pre, target)
    fp = get_FP(pre, target)
    return tp / ( tp + fp + eps )

def Fbeta_Score(pre, target, beta):
    ppv = Precision(pre, target)
    tpr = Recall(pre, target)
    score = ( math.pow(beta, 2) + 1 ) * ppv * tpr / \
        ( math.pow(beta, 2) * ppv + tpr + eps )
    return score

def F1Score(pre, target):
    return Fbeta_Score(pre, target, 1)

def ConsistencyError(pre, target):
    dim = len(target.shape)
    sum_axis = tuple(np.arange(dim - 1))
    diff = pre - target
    diff[ np.where(diff != 0) ] = 1
    c_error = np.sum(diff, sum_axis) / \
        ( np.sum(pre, sum_axis) + eps )
    return c_error

def GlobalConsistencyError(pre, target):
    dim = len(target.shape)
    sum_axis = tuple(np.arange(dim - 1))
    diff = pre - target
    diff[ np.where(diff != 0) ] = 1
    p_error = np.sum(diff, sum_axis) / \
        ( np.sum(pre, sum_axis) + eps )
    t_error = np.sum(diff, sum_axis) / \
        ( np.sum(target, sum_axis) + eps )
    gc_error = np.minimum(p_error, t_error)
    return gc_error

def VolumetricDistance(pre, target):
    dim = len(target.shape)
    sum_axis = tuple(np.arange(dim - 1))
    score = np.abs( np.sum(pre, sum_axis) - np.sum(target, sum_axis) ) / \
        ( np.sum(pre, sum_axis) + np.sum(target, sum_axis) + eps )
    return score

def VolumtricSimilarity(pre, target):
    return 1 - VolumetricDistance(pre, target)

def MutualInformation(pre, target):
    tp = get_TP(pre, target)
    fp = get_FP(pre, target)
    tn = get_TN(pre, target)
    fn = get_FN(pre, target)
    
    n = target.size / target.shape[-1]
    
    p_pro_p = (tp + fn) / n
    p_pro_n = (tn + fn) / n
    
    t_pro_p = (tp + fp) / n
    t_pro_n = (tn + fp) / n
    
    pt_pro_pp = tp / n
    pt_pro_pn = fn / n
    pt_pro_np = fp / n
    pt_pro_nn = tn / n
    
    p_entropy = ( p_pro_p * np.log(p_pro_p + eps) + 
        p_pro_n * np.log(p_pro_n + eps) ) * -1
    
    t_entropy = ( t_pro_p * np.log(t_pro_p + eps) + 
        t_pro_n * np.log(t_pro_n + eps) ) * -1
    
    pt_entropy = ( pt_pro_pp * np.log(pt_pro_pp + eps) + 
        pt_pro_pn * np.log(pt_pro_pn + eps) +
        pt_pro_np * np.log(pt_pro_np + eps) +
        pt_pro_nn * np.log(pt_pro_nn + eps)
        ) * -1
    mi = p_entropy + t_entropy - pt_entropy
    return mi

def VariationofInformation(pre, target):
    tp = get_TP(pre, target)
    fp = get_FP(pre, target)
    tn = get_TN(pre, target)
    fn = get_FN(pre, target)
    
    n = target.size / target.shape[-1]
    
    p_pro_p = (tp + fn) / n
    p_pro_n = (tn + fn) / n
    
    t_pro_p = (tp + fp) / n
    t_pro_n = (tn + fp) / n
    
    pt_pro_pp = tp / n
    pt_pro_pn = fn / n
    pt_pro_np = fp / n
    pt_pro_nn = tn / n
    
    p_entropy = ( p_pro_p * np.log(p_pro_p + eps) + 
        p_pro_n * np.log(p_pro_n + eps) ) * -1
    
    t_entropy = ( t_pro_p * np.log(t_pro_p + eps) + 
        t_pro_n * np.log(t_pro_n + eps) ) * -1
    
    pt_entropy = ( pt_pro_pp * np.log(pt_pro_pp + eps) + 
        pt_pro_pn * np.log(pt_pro_pn + eps) +
        pt_pro_np * np.log(pt_pro_np + eps) +
        pt_pro_nn * np.log(pt_pro_nn + eps)
        ) * -1
    mi = p_entropy + t_entropy - pt_entropy
    voi = p_entropy + t_entropy - 2 * mi
    return voi
    
def GetICCvalue(data):
    n, k, _ = np.shape(data)
    grand_mean = np.mean( data, (0, 1) )
    sst = np.sum( np.power( data - grand_mean, 2), (0, 1) )
    
    ssr = np.sum( np.power(np.mean(data, 1) - grand_mean, 2), 0 ) * k
    ssc = np.sum( np.power(np.mean(data, 0) - grand_mean, 2), 0 ) * n
    
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    
    msw = np.mean( np.var(data, 1) * (k / (k - 1)), 0 )
    return msr, msc, mse, msw
    
def InterclassCorrelation(pre, target, model = 'one-way', type = 'single', absolute = True):
    n = target.size / target.shape[-1]
    k = 2
    tpre = np.reshape(pre, [int(n), target.shape[-1]])
    tpre = np.expand_dims(tpre, 1)
    ttarget = np.reshape(target, [int(n), target.shape[-1]])
    ttarget = np.expand_dims(ttarget, 1)
    
    stat_data = np.concatenate([tpre, ttarget], 1)
    
    msr, msc, mse, msw = GetICCvalue(stat_data)
    bias = 1e-8
    if model == 'one-way':
        if type == 'single':
            return (msr - msw) / (msr + (k - 1) * msw + bias)
        else:
            return (msr - msw) / (msr + bias)
    else:
        if type == 'single':
            if absolute == True:
                return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n + bias)
            else:
                return (msr - mse) / (msr + (k - 1) * mse + bias)
        else:
            if absolute == True:
                return (msr - mse) / (msr + (msc - mse) / n + bias)
            else:
                return (msr - mse) / (msr + bias)

def ProbabilisticDistance(pre, target):
    dim = len(target.shape)
    sum_axis = tuple(np.arange(dim - 1))
    score = np.abs( target - pre ).sum(sum_axis) / ( 2 * (pre * target).sum(sum_axis) + eps )
    return score

def Kappa(pre, target):
    n = target.size / target.shape[-1]
    tp = get_TP(pre, target)
    fp = get_FP(pre, target)
    tn = get_TN(pre, target)
    fn = get_FN(pre, target)
    
    a = tp + tn
    c = ( (tn + fn) * (tn + fp) + (fp + tp) * (fn + tp) ) / n
    score = (a - c) / (n - c + eps)
    return score

def SobelOp(pre, target, mode):
    if mode == '2d':
        sobel_op = SobelTorch.SobelLayer('2d', 1)
        tpre = np.transpose( pre, [2, 0, 1] )
        tpre = np.expand_dims( tpre, 1 )
        tpre = torch.tensor(tpre, dtype = torch.float32)
        
        ttarget = np.transpose( target, [2, 0, 1] )
        ttarget = np.expand_dims( ttarget, 1 )
        ttarget = torch.tensor(ttarget, dtype = torch.float32)
    elif mode == '3d':
        sobel_op = SobelTorch.SobelLayer('3d', 1)
        tpre = np.transpose( pre, [3, 0, 1, 2] )
        tpre = np.expand_dims( tpre, 1 )
        tpre = torch.tensor(tpre, dtype = torch.float32)
        
        ttarget = np.transpose( target, [3, 0, 1, 2] )
        ttarget = np.expand_dims( ttarget, 1 )
        ttarget = torch.tensor(ttarget, dtype = torch.float32)
        
    grad_pre = sobel_op(tpre).detach().numpy()
    grad_target = sobel_op(ttarget).detach().numpy()
    
    length_pre = (np.sqrt( np.square(grad_pre).sum(1) + eps ) > 0.5).astype(np.uint8)
    length_target = (np.sqrt( np.square(grad_target).sum(1) + eps ) > 0.5).astype(np.uint8)
    return length_pre, length_target

def HausdorffDistance(pre, target):
    distance = []
    dim = len(pre.shape)
    if dim == 3:
        mode = '2d'
    elif dim == 4:
        mode = '3d'
    length_pre, length_target = SobelOp(pre, target, mode)
    
    if mode == '2d':
        for i in range(length_pre.shape[0]):
            if np.sum(length_pre[i]) != 0:
                pre_cord = np.nonzero(length_pre[i])
            else:
                if np.sum(pre[..., i]) == 0:
                    pre[pre.shape[0] // 2, pre.shape[1] // 2, i] = 1
                pre_cord = np.nonzero(pre[..., i])
            pre_cord = np.concatenate([
                np.expand_dims(pre_cord[0], 1),
                np.expand_dims(pre_cord[1], 1)
            ], 1)
            pre_cord = np.expand_dims(pre_cord, 1)
            
            if np.sum(length_target[i]) != 0:
                target_cord = np.nonzero(length_target[i])
                target_cord = np.concatenate([
                    np.expand_dims(target_cord[0], 1),
                    np.expand_dims(target_cord[1], 1)
                ], 1)
                target_cord = np.expand_dims(target_cord, 0)
                d = np.sqrt( np.square( target_cord - pre_cord ).sum(2) )
                dpt = d.min(1)
                dtp = d.min(0)
                distance.append( np.concatenate([dpt, dtp]).max() )
    if mode == '3d':
        for i in range(length_pre.shape[0]):
            if np.sum(length_pre[i]) != 0:
                pre_cord = np.nonzero(length_pre[i])
            else:
                if np.sum(pre[..., i]) == 0:
                    pre[pre.shape[0] // 2, pre.shape[1] // 2, i] = 1
                pre_cord = np.nonzero(pre[..., i])
            pre_cord = np.concatenate([
                np.expand_dims(pre_cord[0], 1),
                np.expand_dims(pre_cord[1], 1),
                np.expand_dims(pre_cord[2], 1)
            ], 1)
            pre_cord = np.expand_dims(pre_cord, 1)
            
            if np.sum(length_target[i]) != 0:
                target_cord = np.nonzero(length_target[i])
                target_cord = np.concatenate([
                    np.expand_dims(target_cord[0], 1),
                    np.expand_dims(target_cord[1], 1),
                    np.expand_dims(target_cord[2], 1)
                ], 1)
                target_cord = np.expand_dims(target_cord, 0)
                d = np.sqrt( np.square( target_cord - pre_cord ).sum(2) )
                dpt = d.min(1)
                dtp = d.min(0)
                distance.append( np.concatenate([dpt, dtp]).max() )
    return distance

def HausdorffDistancePercent(pre, target, percent):
    distance = []
    dim = len(pre.shape)
    if dim == 3:
        mode = '2d'
    elif dim == 4:
        mode = '3d'
    length_pre, length_target = SobelOp(pre, target, mode)
    
    if mode == '2d':
        for i in range(length_pre.shape[0]):
            if np.sum(length_pre[i]) != 0:
                pre_cord = np.nonzero(length_pre[i])
            else:
                if np.sum(pre[..., i]) == 0:
                    pre[pre.shape[0] // 2, pre.shape[1] // 2, i] = 1
                pre_cord = np.nonzero(pre[..., i])
            pre_cord = np.concatenate([
                np.expand_dims(pre_cord[0], 1),
                np.expand_dims(pre_cord[1], 1)
            ], 1)
            pre_cord = np.expand_dims(pre_cord, 1)
            
            if np.sum(length_target[i]) != 0:
                target_cord = np.nonzero(length_target[i])
                target_cord = np.concatenate([
                    np.expand_dims(target_cord[0], 1),
                    np.expand_dims(target_cord[1], 1)
                ], 1)
                target_cord = np.expand_dims(target_cord, 0)
                d = np.sqrt( np.square( target_cord - pre_cord ).sum(2) )
                dpt = d.min(1)
                dtp = d.min(0)
                distance.append( np.percentile(np.concatenate([dpt, dtp]), percent).max() )
    if mode == '3d':
        for i in range(length_pre.shape[0]):
            if np.sum(length_pre[i]) != 0:
                pre_cord = np.nonzero(length_pre[i])
            else:
                if np.sum(pre[..., i]) == 0:
                    pre[pre.shape[0] // 2, pre.shape[1] // 2, i] = 1
                pre_cord = np.nonzero(pre[..., i])
            pre_cord = np.concatenate([
                np.expand_dims(pre_cord[0], 1),
                np.expand_dims(pre_cord[1], 1),
                np.expand_dims(pre_cord[2], 1)
            ], 1)
            pre_cord = np.expand_dims(pre_cord, 1)
            
            if np.sum(length_target[i]) != 0:
                target_cord = np.nonzero(length_target[i])
                target_cord = np.concatenate([
                    np.expand_dims(target_cord[0], 1),
                    np.expand_dims(target_cord[1], 1),
                    np.expand_dims(target_cord[2], 1)
                ], 1)
                target_cord = np.expand_dims(target_cord, 0)
                d = np.sqrt( np.square( target_cord - pre_cord ).sum(2) )
                dpt = d.min(1)
                dtp = d.min(0)
                distance.append( np.percentile(np.concatenate([dpt, dtp]), percent).max() )
    return distance

def HausdorffDistance95(pre, target):
    return HausdorffDistancePercent(pre, target, 95)

def AverageHausdorffDistance(pre, target):
    distance = []
    dim = len(pre.shape)
    if dim == 3:
        mode = '2d'
    elif dim == 4:
        mode = '3d'
    length_pre, length_target = SobelOp(pre, target, mode)
    
    if mode == '2d':
        for i in range(length_pre.shape[0]):
            if np.sum(length_pre[i]) != 0:
                pre_cord = np.nonzero(length_pre[i])
            else:
                if np.sum(pre[..., i]) == 0:
                    pre[pre.shape[0] // 2, pre.shape[1] // 2, i] = 1
                pre_cord = np.nonzero(pre[..., i])
            pre_cord = np.concatenate([
                np.expand_dims(pre_cord[0], 1),
                np.expand_dims(pre_cord[1], 1)
            ], 1)
            pre_cord = np.expand_dims(pre_cord, 1)
            
            if np.sum(length_target[i]) != 0:
                target_cord = np.nonzero(length_target[i])
                target_cord = np.concatenate([
                    np.expand_dims(target_cord[0], 1),
                    np.expand_dims(target_cord[1], 1)
                ], 1)
                target_cord = np.expand_dims(target_cord, 0)
                d = np.sqrt( np.square( target_cord - pre_cord ).sum(2) )
                dpt = d.min(1)
                dtp = d.min(0)
                distance.append( np.maximum(dpt.mean(), dtp.mean()) )
    if mode == '3d':
        for i in range(length_pre.shape[0]):
            if np.sum(length_pre[i]) != 0:
                pre_cord = np.nonzero(length_pre[i])
            else:
                if np.sum(pre[..., i]) == 0:
                    pre[pre.shape[0] // 2, pre.shape[1] // 2, i] = 1
                pre_cord = np.nonzero(pre[..., i])
            pre_cord = np.concatenate([
                np.expand_dims(pre_cord[0], 1),
                np.expand_dims(pre_cord[1], 1),
                np.expand_dims(pre_cord[2], 1)
            ], 1)
            pre_cord = np.expand_dims(pre_cord, 1)
            
            if np.sum(length_target[i]) != 0:
                target_cord = np.nonzero(length_target[i])
                target_cord = np.concatenate([
                    np.expand_dims(target_cord[0], 1),
                    np.expand_dims(target_cord[1], 1),
                    np.expand_dims(target_cord[2], 1)
                ], 1)
                target_cord = np.expand_dims(target_cord, 0)
                d = np.sqrt( np.square( target_cord - pre_cord ).sum(2) )
                dpt = d.min(1)
                dtp = d.min(0)
                distance.append( np.maximum(dpt.mean(), dtp.mean()) )
    return distance

def AverageSurfaceDistance(pre, target):
    distance = []
    dim = len(pre.shape)
    if dim == 3:
        mode = '2d'
    elif dim == 4:
        mode = '3d'
    length_pre, length_target = SobelOp(pre, target, mode)
    
    if mode == '2d':
        for i in range(length_pre.shape[0]):
            if np.sum(length_pre[i]) != 0:
                pre_cord = np.nonzero(length_pre[i])
            else:
                if np.sum(pre[..., i]) == 0:
                    pre[pre.shape[0] // 2, pre.shape[1] // 2, i] = 1
                pre_cord = np.nonzero(pre[..., i])
            pre_cord = np.concatenate([
                np.expand_dims(pre_cord[0], 1),
                np.expand_dims(pre_cord[1], 1)
            ], 1)
            pre_cord = np.expand_dims(pre_cord, 1)
            
            if np.sum(length_target[i]) != 0:
                target_cord = np.nonzero(length_target[i])
                target_cord = np.concatenate([
                    np.expand_dims(target_cord[0], 1),
                    np.expand_dims(target_cord[1], 1)
                ], 1)
                target_cord = np.expand_dims(target_cord, 0)
                d = np.sqrt( np.square( target_cord - pre_cord ).sum(2) )
                dpt = d.min(1)
                distance.append( dpt.mean() )
    if mode == '3d':
        for i in range(length_pre.shape[0]):
            if np.sum(length_pre[i]) != 0:
                pre_cord = np.nonzero(length_pre[i])
            else:
                if np.sum(pre[..., i]) == 0:
                    pre[pre.shape[0] // 2, pre.shape[1] // 2, i] = 1
                pre_cord = np.nonzero(pre[..., i])
            pre_cord = np.concatenate([
                np.expand_dims(pre_cord[0], 1),
                np.expand_dims(pre_cord[1], 1),
                np.expand_dims(pre_cord[2], 1)
            ], 1)
            pre_cord = np.expand_dims(pre_cord, 1)
            
            if np.sum(length_target[i]) != 0:
                target_cord = np.nonzero(length_target[i])
                target_cord = np.concatenate([
                    np.expand_dims(target_cord[0], 1),
                    np.expand_dims(target_cord[1], 1),
                    np.expand_dims(target_cord[2], 1)
                ], 1)
                target_cord = np.expand_dims(target_cord, 0)
                d = np.sqrt( np.square( target_cord - pre_cord ).sum(2) )
                dpt = d.min(1)
                distance.append( dpt.mean() )
    return distance

def AverageSymmetricSurfaceDistance(pre, target):
    distance = []
    dim = len(pre.shape)
    if dim == 3:
        mode = '2d'
    elif dim == 4:
        mode = '3d'
    length_pre, length_target = SobelOp(pre, target, mode)
    
    if mode == '2d':
        for i in range(length_pre.shape[0]):
            if np.sum(length_pre[i]) != 0:
                pre_cord = np.nonzero(length_pre[i])
            else:
                if np.sum(pre[..., i]) == 0:
                    pre[pre.shape[0] // 2, pre.shape[1] // 2, i] = 1
                pre_cord = np.nonzero(pre[..., i])
            pre_cord = np.concatenate([
                np.expand_dims(pre_cord[0], 1),
                np.expand_dims(pre_cord[1], 1)
            ], 1)
            pre_cord = np.expand_dims(pre_cord, 1)
            
            if np.sum(length_target[i]) != 0:
                target_cord = np.nonzero(length_target[i])
                target_cord = np.concatenate([
                    np.expand_dims(target_cord[0], 1),
                    np.expand_dims(target_cord[1], 1)
                ], 1)
                target_cord = np.expand_dims(target_cord, 0)
                d = np.sqrt( np.square( target_cord - pre_cord ).sum(2) )
                dpt = d.min(1)
                dtp = d.min(0)
                distance.append( 0.5 * ( dpt.mean() + dtp.mean() ) )
    if mode == '3d':
        for i in range(length_pre.shape[0]):
            if np.sum(length_pre[i]) != 0:
                pre_cord = np.nonzero(length_pre[i])
            else:
                if np.sum(pre[..., i]) == 0:
                    pre[pre.shape[0] // 2, pre.shape[1] // 2, i] = 1
                pre_cord = np.nonzero(pre[..., i])
            pre_cord = np.concatenate([
                np.expand_dims(pre_cord[0], 1),
                np.expand_dims(pre_cord[1], 1),
                np.expand_dims(pre_cord[2], 1)
            ], 1)
            pre_cord = np.expand_dims(pre_cord, 1)
            
            if np.sum(length_target[i]) != 0:
                target_cord = np.nonzero(length_target[i])
                target_cord = np.concatenate([
                    np.expand_dims(target_cord[0], 1),
                    np.expand_dims(target_cord[1], 1),
                    np.expand_dims(target_cord[2], 1)
                ], 1)
                target_cord = np.expand_dims(target_cord, 0)
                d = np.sqrt( np.square( target_cord - pre_cord ).sum(2) )
                dpt = d.min(1)
                dtp = d.min(0)
                distance.append( 0.5 * ( dpt.mean() + dtp.mean() ) )
    return distance

def debug():
    pre = np.zeros([128, 128])
    pre[50: 90, 60: 120] = 1
    pre[10: 30, 20: 50] = 2
    
    target = np.zeros([128, 128])
    target[30: 70, 40: 80] = 1
    target[10: 30, 10: 40] = 2
    
    pre = one_hot(pre, 3)
    print('pre', pre.shape)
    target = one_hot(target, 3)
    print('target', target.shape)           
    metrics = {}
    
    dsc = DiceCoefficient(pre, target)
    metrics['dsc'] = dsc
    jc = JaccardCoefficient(pre, target)
    metrics['jac'] = jc
    sen = Recall(pre, target)
    metrics['sen'] = sen
    spe = Specificity(pre, target)
    metrics['spe'] = spe
    fpr = FPR(pre, target)
    metrics['fpr'] = fpr
    fnr = FNR(pre, target)
    metrics['fnr'] = fnr
    ppv = Precision(pre, target)
    metrics['ppv'] = ppv
    f1 = F1Score(pre, target)
    metrics['f1'] = f1
    gce = GlobalConsistencyError(pre, target)
    metrics['gce'] = gce
    vd = VolumetricDistance(pre, target)
    metrics['vd'] = vd
    vs = VolumtricSimilarity(pre, target)
    metrics['vs'] = vs
    mi = MutualInformation(pre, target)
    metrics['mi'] = mi
    voi = VariationofInformation(pre, target)
    metrics['voi'] = voi
    icc = InterclassCorrelation(pre, target)
    metrics['icc'] = icc
    pbd = ProbabilisticDistance(pre, target)
    metrics['pbd'] = pbd
    kappa = Kappa(pre, target)
    metrics['kappa'] = kappa
    hd = HausdorffDistance(pre, target)
    metrics['hd'] = hd
    hd95 = HausdorffDistancePercent(pre, target, 95)
    metrics['hd95'] = hd95
    avd = AverageHausdorffDistance(pre, target)
    metrics['avd'] = avd   
    assd = AverageSymmetricSurfaceDistance(pre, target)
    metrics['assd'] = assd
    for k, v in metrics.items():
        print(k, v)

if __name__ == '__main__':
    debug()
    