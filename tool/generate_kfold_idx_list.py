# coding:utf-8
"""
 ╦╔═╗╦═╗╦ ╦╔╗╔╔═╗╔═╗╦ ╦
 ║║ ║╠╦╝║║║║║║╠═╝╠═╣╚╦╝
╚╝╚═╝╩╚═╚╩╝╝╚╝╩  ╩ ╩ ╩ 
@time: 2022/04/02
@file: generate_kfold_idx_list.py                
@author: Jorwnpay                    
@contact: jwp@mail.nankai.edu.cn                                         
"""   
import os
import numpy as np
import argparse

def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='baseline')
    parser.add_argument('save_path', type=str)
    parser.add_argument('--dataset', type=str, default='FLSMDD')
    args = parser.parse_args()
    return args

'''
Generate 10-trail 5-fold cross-validation index list 
    @ dataset : name of dataset, can be FLSMDD or NKSID
    @ p : p is repeat time of random k-fold test
    @ k : k is the k param of k-fold
'''
def gen_kfold_txt(dataset, p=10, k=5):
    FLSMDD_class_slices = [0, 449, 816, 1042, 1391, 1524, 1661, 1760, 1825, 2156, 2364]
    NKSID_class_slices = [0, 203, 491, 511, 1462, 1574, 1668, 1783, 2617]
    slices_list = []
    
    # curr_dir = os.path.dirname(__file__)
    kfold_train_path = os.path.join(args.save_path, 'kfold_train.txt') 
    kfold_val_path = os.path.join(args.save_path, 'kfold_val.txt') 
    f_train = open(kfold_train_path, 'a+')
    f_val = open(kfold_val_path, 'a+')
    f_train.truncate()
    f_val.truncate()
    if dataset == 'FLSMDD':
        for i in range(len(FLSMDD_class_slices) - 1):
            arr = np.arange(FLSMDD_class_slices[i], FLSMDD_class_slices[i+1], dtype=np.int32)
            slices_list.append(arr)
    elif dataset == 'NKSID':
        for i in range(len(NKSID_class_slices) - 1):
            arr = np.arange(NKSID_class_slices[i], NKSID_class_slices[i+1], dtype=np.int32)
            slices_list.append(arr)
    else:
        print(f'ERROR! DATASET {dataset} IS NOT EXIST!')

    for ip in range(p):
        kfold_list = []
        for s in range(len(slices_list)):
            np.random.shuffle(slices_list[s])
            arr_len = len(slices_list[s])
            fold_len = arr_len // k
            # print(f'fold_len: {fold_len}')
            if s == 0:
                for ik in range(k):
                    if ik == k-1: # the last fold
                        kfold_list.append(slices_list[s][ik*fold_len : ])
                    else:
                        kfold_list.append(slices_list[s][ik*fold_len : (ik+1)*fold_len])
            else:
                for ik in range(k):
                    if ik == k-1: # the last fold
                        kfold_list[ik] = np.append(kfold_list[ik], slices_list[s][ik*fold_len : ])
                    else:
                        kfold_list[ik] = np.append(kfold_list[ik], slices_list[s][ik*fold_len : (ik+1)*fold_len])
        # print(kfold_list)
        for ik in range(k):
            val_list = kfold_list[ik]
            train_list = np.array([], dtype=np.int32)
            for iik in range(k):
                if iik != ik:
                    train_list = np.append(train_list, kfold_list[iik])
            f_val.write(f'#p{ip}-k{ik}\n')
            for v in val_list:
                f_val.write(str(v) + ' ')
            f_val.write('\n')
            f_train.write(f'#p{ip}-k{ik}\n')
            # f_train.write(str(train_list))
            for t in train_list:
                f_train.write(str(t) + ' ')
            f_train.write('\n')
            # print(f'--------------------{ik}--------------------')
            # print(val_list)
            # print(train_list)
    print(f'The generated 10-trail 5-fold train list file has been saved to {kfold_train_path}.')
    print(f'The generated 10-trail 5-fold validation list file has been saved to {kfold_val_path}.')
        
if __name__ == '__main__':
    args = parse_args()
    assert args.dataset in ['FLSMDD', 'NKSID']
    gen_kfold_txt(args.dataset, p=10, k=5)
