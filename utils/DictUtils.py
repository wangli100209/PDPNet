import sys
sys.path.append('./')
sys.path.append('../')
import os
import numpy as np
import csv

def Make_Dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class DictUtil():
    def __init__(self):
        super(DictUtil,self).__init__()

    def PrintDict(self, tdict, col = 3):
        tstr = ''
        for i, (key, value) in enumerate(tdict.items()):
            if i != 0 and i % col == 0:
                print(tstr)
                tstr = ''
                tstr += key + ': ' + str(round(value, 4)) + '\t'
            else:
                tstr += key + ': ' + str(round(value, 4)) + '\t'
        print(tstr)

    def SumDictList(self, dict_list):
        sum_dict = {}
        for i, tdict in enumerate(dict_list):
            if i == 0:
                sum_dict = tdict.copy()
            else:
                if tdict is not None:
                    for key, value in tdict.items():
                        sum_dict.update({key: sum_dict[key] + value})
        return sum_dict

    def MeanDictList(self, dict_list):
        sum_dict = self.SumDictList(dict_list)
        for key, value in sum_dict.items():
            sum_dict.update({key: value / ( len(dict_list) - len([item for item in dict_list if item is None]) ) })
        return sum_dict
    
    def SaveDictlist2Csv(self, save_path, dict_list, name_list = None):
        Make_Dir(os.path.dirname(save_path))
        if name_list == None:
            name_list = np.arange(len(dict_list))
        csvfile = open(save_path, 'w', newline = '')
        writer = csv.writer(csvfile)
        keylist = ['name']
        for key, _ in dict_list[0].items():
            keylist.append(key)
        writer.writerow(keylist)
        for i, tdict in enumerate(dict_list):
            value_list = [name_list[i]]
            for _, value in tdict.items():
                value_list.append(value)
            writer.writerow(value_list)
                
    def CompareFunction(self, A, B, compare_rule = 'equal greater'):
        if compare_rule == 'equal greater':
            return A >= B
        if compare_rule == 'greater':
            return A > B
        if compare_rule == 'equal less':
            return A<= B
        if compare_rule == 'less':
            return A < B

    def CompareDicts(self, dict_A, dict_B, wights = None, compare_rule = None):
        update_num = 0
        total_num = 0
        for key, value in dict_A.items():
            if wights == None:
                total_num += 1
            else:
                total_num += wights[key]
            if compare_rule != None:
                if self.CompareFunction(dict_A[key], dict_B[key], compare_rule[key]):
                    if wights == None:
                        update_num += 1
                    else:
                        update_num += wights[key]
            else:
                if self.CompareFunction(dict_A[key], dict_B[key], 'equal greater'):
                    if wights == None:
                        update_num += 1
                    else:
                        update_num += wights[key]
        return update_num / total_num

    def ExtendDicts(self, dict_A, dict_B):
        A_keys = dict_A.keys()
        B_keys = dict_B.keys()
        re = dict_A.copy()
        for key in B_keys:
            i = 0
            tkey = key
            while tkey in A_keys:
                i += 1
                tkey = key + '_' + str(i)
            re[tkey] = dict_B[key]
        return re
