# coding: utf-8
"""
 ╦╔═╗╦═╗╦ ╦╔╗╔╔═╗╔═╗╦ ╦
 ║║ ║╠╦╝║║║║║║╠═╝╠═╣╚╦╝
╚╝╚═╝╩╚═╚╩╝╝╚╝╩  ╩ ╩ ╩ 
@time: 2023/04/02
@file: my_utils.py                
@author: Jorwnpay                    
@contact: jwp@mail.nankai.edu.cn                                         
"""  
import os
import random
import time
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from My_Dataset import MyDataset
import numpy as np
from scipy.stats.mstats import gmean
from sklearn.metrics import confusion_matrix
import pretrainedmodels as ptm
from ImbalancedDatasetSampler import ImbalancedDatasetSampler
import pickle
import timm

class FC(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FC, self).__init__()
        self.fc = nn.Linear(num_inputs, num_outputs, bias=True)
    def forward(self, x):
        y = self.fc(x.view(x.shape[0], -1))
        return y

'''
Get fully connected layers
    @ num_in : number of input channels
    @ num_out : number of output channels
    @ num_fcs : number of fully connected layers
'''
def get_fcs(num_in, num_out, num_fcs, device=None):
    fcs = []
    for i in range(num_fcs):
        if device:
            fcs.append(FC(num_in, num_out).to(device))
        else:
            fcs.append(FC(num_in, num_out))
    return fcs

'''
Get pretrained model.
    @ model_name : for example 'resnet18'
    @ is_train : model is using for training or evaluating 
'''
def get_pretrained_model(model_name, is_train=True, device=None, checkpoint=None):
    if 'vit' in model_name:
        model = timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=checkpoint
        )
    else:
        model = ptm.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    if is_train:
        model.train()
    else:
        model.eval() 
    if device:
        model = model.to(device)
    return model
    
'''
Fix the output class number.
    @ model : pretrained model
    @ nb_classes : output class number
'''
def model_fc_fix(model, nb_classes, device=None):
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, nb_classes, device=device)
    return model

'''
Get k-fold image indexes.
    @ p : index of 10-trails
    @ k : index of 5-folds
    @ dataset : name of dataset, can be FLSMDD or NKSID
    @ data_dir: directory of dataset
    @ sample_type: sampling strategy, can be uniform or balance
'''
def get_kfold_img_idx(p, k, dataset, data_dir, sample_type=None):  
    '''
    balance_num: set the balance number of each class in dataset
    class_slices: index range of every class for the dataset, like class 0: [0, 43], class 1: [43, 313]
    '''
    if sample_type == 'balance':
        if 'FLSMDD' in dataset:
            balance_num = 115
            class_slices = [0, 449, 816, 1042, 1391, 1524, 1661, 1760, 1825, 2156, 2364]
        elif 'NKSID' in dataset:
            balance_num = 108
            class_slices = [0, 203, 491, 511, 1462, 1574, 1668, 1783, 2617]
        else: 
            print(f'ERROR! Dataset {dataset} is not exist!')
            return None

    kfold_train_path = os.path.join(data_dir, 'kfold_train.txt') 
    kfold_val_path = os.path.join(data_dir, 'kfold_val.txt') 
    f_train = open(kfold_train_path, 'r')
    f_val = open(kfold_val_path, 'r')
    train_lines = f_train.readlines()
    val_lines = f_val.readlines()
    train_line = None
    val_line = None
    for i in range(len(train_lines)):
        if train_lines[i][0] == '#':
            str_cuts = train_lines[i].rstrip().split('-')
            p_read = int(str_cuts[0][2:])
            k_read = int(str_cuts[1][1:])
            if p_read == p and k_read == k:
                train_line = train_lines[i+1]
                break
    for i in range(len(val_lines)):
        if val_lines[i][0] == '#':
            str_cuts = val_lines[i].rstrip().split('-')
            p_read = int(str_cuts[0][2:])
            k_read = int(str_cuts[1][1:])
            if p_read == p and k_read == k:
                val_line = val_lines[i+1]
                break
    train_line = [int(str) for str in train_line.rstrip().split(' ')]
    val_line = [int(str) for str in val_line.rstrip().split(' ')]
    if sample_type:
        if sample_type == 'uniform':
            train_line = [random.sample(train_line, 1)[0] for n in range(len(train_line))]
        elif sample_type == 'balance':
            balance_train_line = []
            balance_num = 0
            train_line.sort()
            for i in range(len(class_slices) - 1):
                tmp_arr = [tr for tr in train_line if tr >= class_slices[i] and tr < class_slices[i+1]]
                balance_train_line = balance_train_line + [random.sample(tmp_arr, 1)[0] for n in range(balance_num)]
            train_line = balance_train_line
        else:
            print(f'ERROR! sample_type {sample_type} is not exist!')
            return None
    return train_line, val_line

'''
Get open-set k-fold image indexes by split 
    @ dataset : name of dataset, can be FLSMDD or NKSID
    @ train_idx : indexes of training samples, generated by 'get_kfold_img_idx()'
    @ train_class_num : number of training class
    @ test_class_num : number of test class
    @ includes_all_train_class : choose if including all train classes into test class
    @ random_seed : set random seed for sampling training class, we set random_seed=$p, and keep consistent among all compared methods
    @ sample_type : sampling strategy, can be uniform or balance
    @ unknown_label : define the value of unknown label. If unknown_label == None, set unknown_label = train_class_num; or use the given value(e.g., -1)
    @ silence : if print labels' mapping information
'''
def get_open_img_idx(dataset, train_idx, train_class_num=2, test_class_num=3, includes_all_train_class=True, random_seed=0, sample_type='uniform', unknown_label=None, silence=True):
    if dataset == 'FLSMDD':
        class_slices = [0, 449, 816, 1042, 1391, 1524, 1661, 1760, 1825, 2156, 2364]
        balance_num = 115
    elif dataset == 'NKSID':
        class_slices = [0, 203, 491, 511, 1462, 1574, 1668, 1783, 2617]
        balance_num = 108
    else:
        print(f'ERROR! DATASET {dataset} IS NOT EXIST!')
        return None
    all_class_num = len(class_slices) - 1
    assert train_class_num > 0 and train_class_num < all_class_num
    if includes_all_train_class:
        assert test_class_num >= train_class_num  # not include equal to ensure openness.
    rnd = np.random.RandomState(random_seed)  # Ensure identical random index for all compared methods.
    class_list = list(range(all_class_num))
    train_classes = rnd.choice(class_list, train_class_num, replace=False).tolist()
    if includes_all_train_class:
        rnd = np.random.RandomState(random_seed)
        test_classes = rnd.choice(class_list, test_class_num, replace=False).tolist()
    else:
        test_classes = rnd.choice(class_list, test_class_num, replace=False).tolist()

    open_train_idx = []
    # choose open class train indexes from all train indexes
    for i in train_classes:
        tmp_arr = [tr for tr in train_idx if tr >= class_slices[i] and tr < class_slices[i+1]]
        # This model uses 'rnd' to ensure that \ 
        # the training data used between different methods \
        # is completely consistent and random.
        if sample_type == 'uniform':            
            open_train_idx = open_train_idx + rnd.choice(tmp_arr, len(tmp_arr), replace=True).tolist()
        # This model uses 'np.random.choice' to obtain \
        # a random validation set that is different \
        # from the training data
        elif sample_type == 'random_uniform':   
            open_train_idx = open_train_idx + np.random.choice(tmp_arr, len(tmp_arr), replace=True).tolist()
        elif sample_type == 'balance':
            open_train_idx = open_train_idx + np.random.choice(tmp_arr, balance_num, replace=True).tolist()
        else:
            print(f'ERROR! sample_type {sample_type} is not exist!')
            return None
        
    # save the mapping of {old label : new label}
    if not unknown_label: # if unknown_label = None，set unknown_label = train_class_num; or use the given value（e.g., -1）
        unknown_label = train_class_num
    old_new_lbl_map_train = {train_classes[i] : i for i in range(train_class_num)}
    old_new_lbl_map_test = {test_classes[i] : i if i < train_class_num else unknown_label for i in range(test_class_num)}
    
    if not silence:
        print(f'old_new_lbl_map_train: {old_new_lbl_map_train}')
        print(f'old_new_lbl_map_test: {old_new_lbl_map_test}')
    return open_train_idx, old_new_lbl_map_train, old_new_lbl_map_test

'''
Get k-fold image data iters.
    @ batch_size : batch size
    @ data_dir : dataset directory
    @ train_idx : index list(s) of train images you want, like [[1,2,3],[4,5,6],...] or [1,2,3,...]
    @ val_idx : index list(s) of validation images you want
    @ mean : the mean of the ImageNet dataset
    @ std : the variance of the ImageNet dataset  
    @ old_new_lbl_map_train : dictionary of training labels mapping infomation. {'old_label': new_label}, e.g., {'0': 3}
    @ old_new_lbl_map_test : dictionary of test labels mapping infomation
    @ sampler : if using class-balanced sampler, set this item as 'balance' 
'''
def get_kfold_img_iters(batch_size, data_dir, train_idx, val_idx, mean, std, old_new_lbl_map_train=None, old_new_lbl_map_test=None, sampler=None):
    normalize = transforms.Normalize(mean=mean, 
                                     std=std)
    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),   
        transforms.RandomRotation(90),       
        transforms.ToTensor(),
        normalize                  
    ])
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize        
    ])

    if isinstance(train_idx[0], list):
        train_iter = []
        for idx in train_idx:   
            train_imgs = MyDataset(os.path.join(data_dir, 'train_abs.txt'), idx, 
                                transform=train_transform, old_new_lbl_map=old_new_lbl_map_train)
            if sampler == 'balance':
                samp = ImbalancedDatasetSampler(train_imgs)
                shuf = False
            else:
                samp = None
                shuf = True
            train_iter.append(torch.utils.data.DataLoader(train_imgs, 
                                                batch_size=batch_size, 
                                                sampler=samp,
                                                shuffle=shuf,     
                                                num_workers=4))  
    elif isinstance(train_idx[0], int):
        train_imgs = MyDataset(os.path.join(data_dir, 'train_abs.txt'), train_idx, 
                                transform=train_transform, old_new_lbl_map=old_new_lbl_map_train)
        if sampler == 'balance':
            samp = ImbalancedDatasetSampler(train_imgs)
            shuf = False
        else:
            samp = None
            shuf = True
        train_iter = torch.utils.data.DataLoader(train_imgs, 
                                            batch_size=batch_size, 
                                            sampler=samp,
                                            shuffle=shuf,           
                                            num_workers=4)      
        
    val_imgs = MyDataset(os.path.join(data_dir, 'train_abs.txt'), val_idx, 
                                transform=val_transform, old_new_lbl_map=old_new_lbl_map_test)
    val_iter = torch.utils.data.DataLoader(val_imgs, 
                                            batch_size=batch_size, 
                                            num_workers=4)
    return train_iter, val_iter

'''
Get k-fold image data iters for proser method.
    @ batch_size : batch size
    @ data_dir : dataset directory
    @ train_idx : index list(s) of train images you want, like [[1,2,3],[4,5,6],...] or [1,2,3,...]
    @ val_idx : index list(s) of validation images you want
    @ mean : the mean of the ImageNet dataset
    @ std : the variance of the ImageNet dataset  
    @ old_new_lbl_map_train : dictionary of training labels mapping infomation. {'old_label': new_label}, e.g., {'0': 3}
    @ old_new_lbl_map_test : dictionary of test labels mapping infomation.
    @ unk_lbl : define the value of unknown label. If unknown_label == None, set unknown_label = train_class_num; or use the given value(e.g., -1)
'''
def get_kfold_img_iters_proser(batch_size, data_dir, train_idx, val_idx, mean, std, old_new_lbl_map_train=None, old_new_lbl_map_test=None, unk_lbl=None):
    normalize = transforms.Normalize(mean=mean, 
                                     std=std)
    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.Grayscale(num_output_channels=1),    
        transforms.RandomHorizontalFlip(),   
        transforms.RandomRotation(90),       
        transforms.ToTensor(),
        normalize                         
    ])
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize                           
    ])

    if isinstance(train_idx[0], list):
        train_iter = []
        for idx in train_idx:   
            train_imgs = MyDataset(os.path.join(data_dir, 'train_abs.txt'), idx, 
                                transform=train_transform, old_new_lbl_map=old_new_lbl_map_train)
            train_iter.append(torch.utils.data.DataLoader(train_imgs, 
                                                batch_size=batch_size, 
                                                shuffle=True,       
                                                num_workers=4))    
    elif isinstance(train_idx[0], int):
        train_imgs = MyDataset(os.path.join(data_dir, 'train_abs.txt'), train_idx, 
                                transform=train_transform, old_new_lbl_map=old_new_lbl_map_train)
        train_iter = torch.utils.data.DataLoader(train_imgs, 
                                            batch_size=batch_size, 
                                            shuffle=True,          
                                            num_workers=4)       
        
    val_imgs = MyDataset(os.path.join(data_dir, 'train_abs.txt'), val_idx, 
                                transform=val_transform, old_new_lbl_map=old_new_lbl_map_test)
    val_labels = np.array([i[-1] for i in val_imgs])
    unk_val_idx = np.array(val_idx)[val_labels==unk_lbl]   
    kn_val_idx = np.array(val_idx)[val_labels!=unk_lbl]    
    unk_val_imgs = MyDataset(os.path.join(data_dir, 'train_abs.txt'), unk_val_idx, 
                                transform=val_transform, old_new_lbl_map=old_new_lbl_map_test)
    kn_val_imgs = MyDataset(os.path.join(data_dir, 'train_abs.txt'), kn_val_idx, 
                                transform=val_transform, old_new_lbl_map=old_new_lbl_map_test)
    unk_val_iter = torch.utils.data.DataLoader(unk_val_imgs, 
                                                batch_size=batch_size, 
                                                num_workers=4)
    kn_val_iter = torch.utils.data.DataLoader(kn_val_imgs, 
                                                batch_size=batch_size, 
                                                num_workers=4)
    return train_iter, kn_val_iter, unk_val_iter, kn_val_imgs, unk_val_imgs

'''
Evaluate models via accuracy metric.
    @ data_iter : validation data iter
    @ net : model to be evaluate
    @ device : inference device
'''
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

'''
Regular train function.
    @ train_iter : train data iter
    @ test_iter : test data iter
    @ net : model to be trained
    @ loss : loss function
    @ optimizer : training optimizer
    @ device : training device
    @ num_epochs : training epochs
'''
def train_plain(train_iter, test_iter, net, loss, optimizer, scheduler, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y) 
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        scheduler.step()
        
        print('epoch %d, loss %.4f, lr %.4f train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, optimizer.state_dict()['param_groups'][0]['lr'], train_acc_sum / n, time.time() - start))

'''
Balanced muti-branch training function.
    @ train_iter : train data iter
    @ backbone : pretrained backbone of the transfer learning phase
    @ fc : fully connected layer 
    @ loss : loss function
    @ optimizer : training optimizer
    @ device : training device
    @ num_epochs : training epochs
'''
def train_mul(train_iter, backbone, fc, loss, optimizer, scheduler, device, num_epochs):
    backbone = backbone.to(device)
    fc = fc.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        backbone.eval() # to make sure backbone is not in 'train' mode 
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            features = backbone(X)
            y_hat = fc(features)
            l = loss(y_hat, y) 
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        scheduler.step()
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))    

'''
Calculate y_hat for one batch data
    @ fcs : a list of fully connected layers to be evaluated 
    @ features : features output by the backbone model
    @ method : fusion strategy, can be normal, hard_voting, soft_voting or weight_averaging 
    @ device : inference device
    @ fusion_net : an abandoned function, you can overlook this parameter
    @ if_get_logits : if you want to get logits output, default is false
'''
def cal_y_hat(fcs, features, method, device, fusion_net = None, if_get_logits=False):
    y_hat = torch.tensor([]).to(device)
    if method == 'normal':
        logits = []
        for i in range(0,len(fcs)): 
            if isinstance(fcs[i], torch.nn.Module):
                fcs[i].eval() 
                logits.append(fcs[i](features).argmax(dim=1).tolist())
        y_hat = torch.tensor(logits).to(device)
    elif method == 'hard_voting':
        labels_sum = torch.tensor([0]).to(device) 
        for i in range(0,len(fcs)):
            if isinstance(fcs[i], torch.nn.Module):
                fcs[i].eval() 
                logits = fcs[i](features)
                labels_sum = labels_sum + F.one_hot(logits.argmax(dim=1), logits.shape[1]) 
        y_hat = labels_sum.argmax(dim=1)
    elif method == 'soft_voting':
        logits_sum = torch.tensor([0]).to(device) 
        for i in range(0,len(fcs)):
            if isinstance(fcs[i], torch.nn.Module):
                fcs[i].eval() 
                logits_sum = logits_sum + fcs[i](features) 
        y_hat = logits_sum.argmax(dim=1)
    elif method == 'stacking':
        if fusion_net == None:
            print('ERROR! FUSION NET SHOULD NOT BE NONE!')
            return None
        logits = torch.tensor([]).to(device)
        for fc in fcs:
            fc.eval()  # open evaluate mode
            logits = torch.cat((logits, fc(features)), 1) 
        fusion_net.eval() # open evaluate mode
        y_hat = fusion_net(logits).argmax(dim=1)
    elif method == 'weight_averaging':
        fus_param = copy.deepcopy(fcs[0].state_dict()) # fusion params of fcs
        for i in range(1, len(fcs)):
            fus_param['fc.weight'] += fcs[i].state_dict()['fc.weight']
            fus_param['fc.bias'] += fcs[i].state_dict()['fc.bias']
        fus_param['fc.weight'] /= len(fcs)
        fus_param['fc.bias'] /= len(fcs)
        fus_fc = copy.deepcopy(fcs[0]) # use deepcopy to incase fcs[0] be changed
        fus_fc.load_state_dict(fus_param)
        y_hat = fus_fc(features).argmax(dim=1)
        if if_get_logits:
            logits = fus_fc(features)
            return y_hat, logits
    else:
        print(f'ERROR! Method {method} is not exist!')
        return None
    return y_hat

'''
Calculate Gmean
    @ y_true : The true label vector, which should be shaped as : [0, 1, 2, 1...]
    @ y_pred : The predictive label vector, which should be shaped as : [0, 2, 1, 1...]
'''
def cal_gmean(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu()
    conf_matrix = confusion_matrix(y_true, y_pred)
    diag = np.diagonal(conf_matrix) # right classifying number per class
    n_per_class = np.sum(conf_matrix, axis=1) # original number per class
    if(0 not in n_per_class):
        acc_per_class = diag / n_per_class
        gm = gmean(acc_per_class)
        return gm
    else:
        print('ERROR: Number of classes can not be 0 !')
        return None

'''
Calculate Open-Set F Measure
    @ y_true : The true label vector, which should be shaped as : [0, 1, 2, 1...]
    @ y_pred : The predictive label vector, which should be shaped as : [0, 2, 1, 1...]
'''
def cal_osfm(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu()
    conf_matrix = confusion_matrix(y_true, y_pred)
    pred_sum = np.sum(conf_matrix, axis=0)
    true_sum = np.sum(conf_matrix, axis=1)
    # print(conf_matrix)
    known_class_num = conf_matrix.shape[0] - 1
    assert known_class_num > 0
    precision_sum = 0
    recall_sum = 0
    for i in range(known_class_num):
        TP = conf_matrix[i][i]
        assert pred_sum[i] > 0
        precision_sum += TP / pred_sum[i]
        assert true_sum[i] > 0
        recall_sum += TP / true_sum[i]
    precision_avg = precision_sum / known_class_num
    recall_avg = recall_sum / known_class_num
    osfm = (2 * precision_avg * recall_avg) / (precision_avg + recall_avg)
    # print(f'my macro-f1 is {osfm}')
    return osfm
    
'''
The function is using for evaluate the gmean result of the baseline (deep transfer learning) method
    @ data_iter: iter of test/evaluate dataset.
    @ net : model to be evaluated.
    @ device : inference device
    @ if_get_y : if you want to get y_hat and y_true output, default is false
    @ if_get_logits : if you want to get logits output, default is false
'''
def evaluate_gmean(data_iter, net, device=None, if_get_y=False, if_get_logits=False):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device 
    y_true = torch.tensor([]).to(device)
    y_hat = torch.tensor([]).to(device)
    logits = torch.tensor([]).to(device)
    with torch.no_grad():
        for X, y in data_iter: 
            X, y = X.to(device), y.to(device)
            y_true = torch.cat((y_true, y), 0)
            if isinstance(net, torch.nn.Module): # get features first
                net.eval() # NOTE: net should NOT be 'train()' here
                logits = torch.cat((logits, net(X)), 0)
                y_hat = torch.cat((y_hat, net(X).argmax(dim=1)), 0)
        # print(y_hat)
        gmean_fc = cal_gmean(y_true, y_hat)
    if if_get_logits:
        return gmean_fc, y_hat, y_true, logits
    if if_get_y:
        return gmean_fc, y_hat, y_true
    return gmean_fc   

'''
The function is using for evaluate the gmean result of BETL method
    @ data_iter: iter of test/evaluate dataset.
    @ backbone: model without the last linear.
    @ fcs: a list of full connected layers.
    @ method: method of how to fusion these full connected layers, 
            could be: 'normal', 'hard_voting', 'soft_voting', and 'weight_averaging'.
    @ fusion_net: an abandoned function, you can overlook this parameter.
    @ device : inference device
    @ if_get_y : if you want to get y_hat and y_true output, default is false
    @ if_get_logits : if you want to get logits output, default is false
'''
def evaluate_gmean_optional(data_iter, backbone, fcs, method, fusion_net=None, device=None, if_get_y=False, if_get_logits=False):
    if device is None and isinstance(backbone, torch.nn.Module):
        device = list(backbone.parameters())[0].device 
    y_hat = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    logits = torch.tensor([]).to(device)
    cat_dim = 1 if method == 'normal' else 0
    with torch.no_grad():
        for X, y in data_iter: 
            X, y = X.to(device), y.to(device)
            y_true = torch.cat((y_true, y), 0)
            backbone.eval() # open evaluate mode
            features = backbone(X)
            y_hat = torch.cat((y_hat, cal_y_hat(fcs, features, method, device, fusion_net)), cat_dim)
            if method == 'weight_averaging' and if_get_logits:
                _, lo = cal_y_hat(fcs, features, method, device, fusion_net, if_get_logits=True)
                logits = torch.cat((logits, lo), cat_dim)
        if method == 'normal':
            gmean = [cal_gmean(y_true, y_hat[i]) for i in range(len(y_hat))]
        else:
            gmean = cal_gmean(y_true, y_hat)
    if method == 'weight_averaging' and if_get_logits:
        return gmean, y_hat, y_true, logits
    if if_get_y:
        return gmean, y_hat, y_true
    return gmean

'''
The function is using for evaluate the overall accuracy result of BETL method
    @ data_iter: iter of test/evaluate dataset.
    @ backbone: model without the last linear.
    @ fcs: a list of full connected layers.
    @ method: method of how to fusion these full connected layers, 
            could be: 'hard_voting', 'soft_voting', 'stacking'.
    @ fusion_net: an abandoned function, you can overlook this parameter.
'''
def evaluate_acc_optional(data_iter, backbone, fcs, method, fusion_net=None, device=None):
    if device is None and isinstance(backbone, torch.nn.Module):
        device = list(backbone.parameters())[0].device 
    y_hat = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    with torch.no_grad():
        for X, y in data_iter: 
            X, y = X.to(device), y.to(device)
            y_true = torch.cat((y_true, y), 0)
            backbone.eval() # open evaluate mode
            features = backbone(X)
            y_hat = torch.cat((y_hat, cal_y_hat(fcs, features, method, device, fusion_net)), 0)
        acc_sum = (y_hat == y_true.to(device)).float().sum().cpu().item()
        acc_ave = acc_sum / len(y_true)
    return acc_ave

'''
The function is using for ensemble pruning in the third phase of BETL
    @ train_iter: iter of training dataset.
    @ backbone: model without the last linear.
    @ fcs: a list of full connected layers.
    @ _p: pruning proportion
    @ use_single_metric: using single metric to calculate the order of fcs, could be 'gmean' or 'acc'.
'''
def ensemble_pruning(train_iter, backbone, fcs, _p=0.6, use_single_metric=None):
    gmean_list =[]
    acc_list = []
    for fc in fcs:
        gmean = evaluate_gmean(train_iter, nn.Sequential(backbone, fc))
        acc = evaluate_accuracy(train_iter, nn.Sequential(backbone, fc))
        print(gmean, acc)
        gmean_list.append(gmean)
        acc_list.append(acc)
    gmean_list = np.array(gmean_list)
    acc_list = np.array(acc_list)
    gmean_sort = np.argsort(-gmean_list)
    acc_sort = np.argsort(-acc_list)
    if use_single_metric is 'gmean':
        return gmean_sort[0:int(len(gmean_sort)*_p)]
    elif use_single_metric is 'acc':
        return acc_sort[0:int(len(acc_sort)*_p)]
    gmean_order = [0 for i in range(len(fcs))]
    acc_order = [0 for i in range(len(fcs))]
    for i in range(len(fcs)):
        gmean_order[gmean_sort[i]] = i
        acc_order[acc_sort[i]] = i
    overall_order = []
    for i in range(len(fcs)):
        overall_order.append((gmean_order[i] + acc_order[i])/2)
    overall_order = np.array(overall_order)
    overall_sort = np.argsort(overall_order)
    return overall_sort[0:int(len(overall_sort)*_p)]

'''
Get a fusion fully connected layer by weight averaging from a list of full connected layers
    @ fcs: a list of full connected layers.
'''
def get_fusion_fc(fcs):
    fus_param = copy.deepcopy(fcs[0].state_dict()) # fusion params of fcs
    for i in range(1, len(fcs)):
        fus_param['fc.weight'] += fcs[i].state_dict()['fc.weight']
        fus_param['fc.bias'] += fcs[i].state_dict()['fc.bias']
    fus_param['fc.weight'] /= len(fcs)
    fus_param['fc.bias'] /= len(fcs)
    fus_fc = copy.deepcopy(fcs[0]) # use deepcopy to incase fcs[0] be changed
    fus_fc.load_state_dict(fus_param)
    fus_fc.eval()
    return fus_fc

'''
Write y_hat, y_true, logits results to file
    @ file_dir: directory of save folder
    @ y_hat: the predict labels of classification model
    @ y_true: the ground truth labels of validation data
    @ logits: the predict scores of classification model
    @ close_y_hat: the predict scores of closed-set evaluation
    @ p : index of 10-trails
    @ k : index of 5-folds
'''
def write_result_to_file(file_dir, y_hat, y_true, logits=None, close_y_hat=None, p=0, k=0):
    y_hat_path = os.path.join(file_dir, 'y_hat.txt')
    y_true_path = os.path.join(file_dir, 'y_true.txt')
    f_y_hat = open(y_hat_path, 'a')
    f_y_true = open(y_true_path, 'a')
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    f_y_hat.write(f'#p{p}-k{k}\n')
    for y_h in y_hat:
        if isinstance(y_h, torch.Tensor):
            f_y_hat.write(str(y_h.astype(np.int32)) + ' ')
        else:
            f_y_hat.write(str(y_h) + ' ')
    f_y_hat.write('\n')

    f_y_true.write(f'#p{p}-k{k}\n')
    for y_t in y_true:
        if isinstance(y_t, torch.Tensor):
            f_y_true.write(str(y_t.astype(np.int32)) + ' ')
        else:
            f_y_true.write(str(y_t) + ' ')
    f_y_true.write('\n')

    if (close_y_hat != None) and len(close_y_hat):
        close_y_hat_path = os.path.join(file_dir, 'close_y_hat.txt')
        f_close_y_hat = open(close_y_hat_path, 'a')
        if isinstance(close_y_hat, torch.Tensor):
            close_y_hat = close_y_hat.cpu().numpy()
        f_close_y_hat.write(f'#p{p}-k{k}\n')
        for c_y_h in close_y_hat:
            if isinstance(c_y_h, torch.Tensor):
                f_close_y_hat.write(str(c_y_h.astype(np.int32)) + ' ')
            else:
                f_close_y_hat.write(str(c_y_h) + ' ')
        f_close_y_hat.write('\n')
        
    if (logits != None) and len(logits):
        logits_path = os.path.join(file_dir, 'logits.txt')
        f_logits = open(logits_path, 'a')
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        f_logits.write(f'#p{p}-k{k}\n')
        logits = np.around(logits, 2)
        for lo in logits:
            for l in lo:
                f_logits.write(str(l) + ' ')
            f_logits.write(',')
        f_logits.write('\n')
    
    print(f'Write result to file {file_dir} finished!')

'''
Read y_hat, y_true, logits results from file
    @ file_dir: directory of save folder
    @ p : index of 5-trails
    @ k : index of 5-folds
    @ if_get_logits : if you want to read logits, default is false
    @ if_get_close_y_hat : if you want to read closed-set preds, default is false
'''
def read_result_from_file(file_dir, p, k, if_get_logits=False, if_get_close_y_hat=False):
    y_hat_path = os.path.join(file_dir, 'y_hat.txt')
    y_true_path = os.path.join(file_dir, 'y_true.txt')
    
    f_y_hat = open(y_hat_path, 'r')
    f_y_true = open(y_true_path, 'r')
    
    y_hat_lines = f_y_hat.readlines()
    y_true_lines = f_y_true.readlines()

    def get_given_p_k_line(lines):
        for i in range(len(lines)):
            if lines[i][0] == '#':
                str_cuts = lines[i].rstrip().split('-')
                p_read = int(str_cuts[0][2:])
                k_read = int(str_cuts[1][1:])
                if p_read == p and k_read == k:
                    line = lines[i+1]
                    return line
        print('ERROR! The given p-trail and k-fold data is not exist!')
        return None
    
    y_hat_line = get_given_p_k_line(y_hat_lines)
    y_true_line = get_given_p_k_line(y_true_lines)
    y_hat_line = [int(str) for str in y_hat_line.rstrip().split(' ')]
    y_true_line = [int(str) for str in y_true_line.rstrip().split(' ')]
    
    if if_get_logits:
        logits_path = os.path.join(file_dir, 'logits.txt')
        f_logits = open(logits_path, 'r')
        logits_lines = f_logits.readlines()
        logits_line = get_given_p_k_line(logits_lines)
        logits_line = [[float(s) for s in str.rstrip().split(' ')] for str in logits_line.rstrip().rstrip(',').split(',')]
    if if_get_close_y_hat:
        close_y_hat_path = os.path.join(file_dir, 'close_y_hat.txt')
        f_close_y_hat = open(close_y_hat_path, 'r')
        close_y_hat_lines = f_close_y_hat.readlines()
        close_y_hat_line = get_given_p_k_line(close_y_hat_lines)
        close_y_hat_line = [float(str) for str in close_y_hat_line.rstrip().split(' ')]

    if if_get_logits and if_get_close_y_hat:
        return y_hat_line, y_true_line, logits_line, close_y_hat_line
    elif if_get_logits:
        return y_hat_line, y_true_line, logits_line
    elif if_get_close_y_hat:
        return y_hat_line, y_true_line, close_y_hat_line
    return y_hat_line, y_true_line

'''
Save a tensor as a image
    @ img_tensor: tensor type data 
    @ save_dir : image saving directory  
'''
def save_tensor_as_img(img_tensor, save_dir):
    from  torchvision import utils as vutils
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(img_tensor.shape[0]):
        img_dir = save_dir + f'/img_{i}.jpg'
        vutils.save_image(img_tensor[i], img_dir, normalize=True)

'''
Write data to a pikle file
    @ file_dir: file save directory
    @ data : data you want to save
    @ p : index of 5-trails
    @ k : index of 5-folds
'''
def write_data_to_pkl(file_dir, data, p=0, k=0):
    assert os.path.isdir(file_dir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, f'p{p}-k{k}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Write data to file {file_path} finished!')

'''
Read data from a pikle file
    @ file_dir: pikle file directory
    @ p : index of 5-trails
    @ k : index of 5-folds
'''
def read_data_from_pkl(file_dir, p=0, k=0):
    assert os.path.isdir(file_dir)
    file_path = os.path.join(file_dir, f'p{p}-k{k}.pkl')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f'Load data from file {file_path} finished!')
    return data