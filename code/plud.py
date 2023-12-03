# coding: utf-8
"""
 ╦╔═╗╦═╗╦ ╦╔╗╔╔═╗╔═╗╦ ╦
 ║║ ║╠╦╝║║║║║║╠═╝╠═╣╚╦╝
╚╝╚═╝╩╚═╚╩╝╝╚╝╩  ╩ ╩ ╩ 
@time: 2023/04/02
@file: plud.py                
@author: Jorwnpay                    
@contact: jwp@mail.nankai.edu.cn                                         
"""  
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import random
from my_utils import *

def push_logit_loss(logit, gamma=1.):
    sigmoid = nn.Sigmoid()
    loss = sigmoid(gamma * logit)
    return loss.mean()

def train_plud_one_epoch(args, epoch, net, train_iter, criterion, optimizer, alpha, gamma, device='cuda'):
    net.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_iter):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        halflenth = int(len(inputs)/2)        
        if halflenth <= 0: continue          
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
    
        prehalfinputs = inputs[:halflenth]   
        prehalflabels = targets[:halflenth]       
        laterhalfinputs = inputs[halflenth:]     
        laterhalflabels = targets[halflenth:]    

        pre_feats = get_feature_map(args, net, prehalfinputs)
        index = torch.randperm(prehalflabels.size(0)).cuda()
        mixed_prefeats = beta * pre_feats + (1 - beta) * pre_feats[index]
        prehalf_close_logits = featmap2clf(args, net, mixed_prefeats)
        
        lateroutputs = net(laterhalfinputs)                          
        right_close_logit = torch.zeros_like(laterhalflabels, dtype=torch.float)
        wrong_close_logits = lateroutputs.clone()
        for i in range(len(laterhalflabels)):
            right_close_logit[i] = lateroutputs[i][laterhalflabels[i]]
            wrong_close_logits[i][laterhalflabels[i]] = -1e9
        
        loss1 = push_logit_loss(prehalf_close_logits, gamma)
        loss2 = criterion(lateroutputs, laterhalflabels) 
        loss3 = push_logit_loss(wrong_close_logits, gamma)
        loss4 = push_logit_loss(-right_close_logit, gamma)
        loss = loss1 + loss2 + loss3 + loss4  
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print(f'L1: {loss1}, L2: {loss2}, L3:{loss3}, L4:{loss4}') 
    print(f'----- Epoch {epoch} train loss is {train_loss} -----')
       
def train_plud(args, net, train_iter, criterion, optimizer, scheduler, device='cuda'): 
    net = net.to(device)
    FineTune_MAX_EPOCH = args.es
    alpha = args.alpha
    gamma = args.gamma
    for epoch in range(FineTune_MAX_EPOCH):
        train_plud_one_epoch(args, epoch, net, train_iter, criterion, optimizer, alpha, gamma, device=device)
        scheduler.step()

def test_plud(net, testloader):
    net.eval()
    scores, y_true = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):  
            inputs, targets = inputs.to(device), targets.to(device)
            y_true.append(targets)
            logits = net(inputs)                        
            scores.append(logits)  
    scores = torch.cat(scores,dim=0).cpu().numpy().tolist()
    y_true = torch.cat(y_true,dim=0).cpu().numpy().tolist()
    return scores, y_true

def get_feature_map(args, net, x):
    if  args.backbone=="WideResnet":
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
        out = net.layer3(out)
        out = F.relu(net.bn1(out))
        return out
    elif args.backbone == 'resnet18':
        out = net.relu(net.bn1(net.conv1(x)))
        out = net.maxpool(out)
        out = net.layer1(out)
        out = net.layer2(out)
        out = net.layer3(out)
        out = net.layer4(out)
        return out
    elif 'vit' in args.backbone:
        out = net.forward_features(x)
        out = net.forward_head(out, pre_logits=True)
        return out
    else:
        print(f'ERROR! Backbone {backbone} is not exist!')
        return None

def featmap2clf(args, net, x):
    if  args.backbone=="WideResnet":
        out = F.avg_pool2d(x, 8)
        out = out.view(out.size(0), -1)
        out = net.linear(out)
        return out
    elif args.backbone == 'resnet18':
        out = net.avgpool(x)
        out = out.view(out.size(0), -1)
        out = net.last_linear(out)
        return out
    elif 'vit' in args.backbone:
        out = net.head(x)
        return out
    else:
        print(f'ERROR! Backbone {backbone} is not exist!')
        return None

def init_random_seeds(seed=0):
    # Fix Random Seed for Stable Training
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    # Set arg parser
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset', default='NKSID', type=str, help='the dataset name')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--backbone', default='resnet18', type=str, help='choosing network')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--es', default=100, type=int, help='epoch size')
    parser.add_argument('--p_value', default=0, type=int, help='trail number of 5-trail 5-fold cross-validation')
    parser.add_argument('--k_value', default=0, type=int, help='fold number of 5-trail 5-fold cross-validation')
    parser.add_argument('--train_class_num', default=5, type=int, help='Number of class used in training')
    parser.add_argument('--test_class_num', default=8, type=int, help='Number of class used in testing')
    parser.add_argument('--includes_all_train_class', default=True,  action='store_true',
                        help='If required all known classes included in testing')
    parser.add_argument('--method_name', default='plud', type=str, help='Method Name')
    parser.add_argument('--alpha', default=1.5, type=float,help='alpha value for beta distribution')
    parser.add_argument('--gamma', default=0.5, type=float,help='gamma value for plud loss')
    parser.add_argument('--save_results', default='True', type=str,
                        help='If save evaluation results to file')
    parser.add_argument('--save_models', default='False', type=str,
                        help='If save evaluation results to file')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    # Parsing args
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Loading data
    print('==> Preparing data..')
    p_v, k_v = args.p_value, args.k_value
    random_seed = p_v
    batch_size = args.bs
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    data_dir = os.path.join('../data', args.dataset) 
    kfold_train_idx, kfold_test_idx = get_kfold_img_idx(p=p_v, k=k_v, dataset=args.dataset, data_dir=data_dir, sample_type=None)
    open_kfold_train_idx, old_new_lbl_map_train, old_new_lbl_map_val = get_open_img_idx(args.dataset, kfold_train_idx, args.train_class_num, args.test_class_num, random_seed=random_seed)
    open_kfold_val_idx, _, _ = get_open_img_idx(args.dataset, kfold_train_idx, args.train_class_num, args.test_class_num, random_seed=random_seed, sample_type='random_uniform', silence=False)
    train_iter, test_iter = get_kfold_img_iters(batch_size, 
                                                data_dir, 
                                                open_kfold_train_idx, 
                                                kfold_test_idx, 
                                                mean, 
                                                std, 
                                                old_new_lbl_map_train=old_new_lbl_map_train, 
                                                old_new_lbl_map_test=old_new_lbl_map_val)
    _, val_iter = get_kfold_img_iters(batch_size, 
                                      data_dir, 
                                      open_kfold_train_idx, 
                                      open_kfold_val_idx, 
                                      mean, 
                                      std, 
                                      old_new_lbl_map_train=old_new_lbl_map_train, 
                                      old_new_lbl_map_test=old_new_lbl_map_val)     

    # Building model
    print('==> Building model..')
    is_train = True
    backbone = args.backbone
    try:
        net = get_pretrained_model(backbone, is_train)
    except:
        print(f'THE INPUT BACKBONE {backbone} IS NOT EXIST!')
    net = model_fc_fix(net, args.train_class_num)

    # Train Settings
    criterion = nn.CrossEntropyLoss()
    fc_params = list(map(id, net.last_linear.parameters()))
    feature_params = filter(lambda p: id(p) not in fc_params, net.parameters())
    optimizer = optim.SGD([{'params': feature_params},
                        {'params': net.last_linear.parameters(), 'lr': args.lr * 10},],
                        lr=args.lr, weight_decay=0.001)
    scheduler = MultiStepLR(optimizer, [35, 75], gamma=0.5)

    # Fix Random Seed for Stable Training
    init_random_seeds()

    # Training
    train_plud(args, net, train_iter, criterion, optimizer, scheduler, device=device)

    # Evaluation
    val_scores, val_y_true = test_plud(net, val_iter)
    scores, y_true = test_plud(net, test_iter)

    # save results
    data_rename = f'{args.dataset}_tr{args.train_class_num}_te{args.test_class_num}'
    if args.save_results in ['True', 'true', True]:
        curr_dir = os.path.dirname(__file__)
        save_dir = os.path.join(curr_dir, '../output/result', data_rename, f'{args.method_name}', backbone)                                       
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)                                             
        write_result_to_file(save_dir, y_true, y_true, scores, p=p_v, k=k_v)

        save_dir_val = os.path.join(curr_dir, '../output/result', data_rename, f'{args.method_name}_val', backbone)
        if not os.path.exists(save_dir_val):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            os.makedirs(save_dir_val)
        write_result_to_file(save_dir_val, val_y_true, val_y_true, val_scores, p=p_v, k=k_v)
    
    if args.save_models in ['True', 'true']:
        curr_dir = os.path.dirname(__file__)
        model_dir = os.path.join(curr_dir, f'../output/model/{data_rename}/')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = model_dir + f'p{p_v}_k{k_v}_{backbone}_{args.method_name}.pth'
        torch.save(net.state_dict(), model_dir)
        print(f'Model saved to {model_dir} successfully!')