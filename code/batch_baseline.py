# coding: utf-8
"""
 ╦╔═╗╦═╗╦ ╦╔╗╔╔═╗╔═╗╦ ╦
 ║║ ║╠╦╝║║║║║║╠═╝╠═╣╚╦╝
╚╝╚═╝╩╚═╚╩╝╝╚╝╩  ╩ ╩ ╩ 
@time: 2023/04/02
@file: batch_baseline.py                
@author: Jorwnpay                    
@contact: jwp@mail.nankai.edu.cn                                         
"""  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
import argparse
from openmax import compute_train_scores_mavs_dists, fit_weibull, openmax
from my_utils import *

def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='PyTorch Sonar Image Training')
    parser.add_argument('--dataset', default='SCTD', type=str, help='the dataset name')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--backbone', default='resnet18', type=str, help='choosing network')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--es', default=5, type=int, help='epoch size')
    parser.add_argument('--p_value', default=0, type=int, help='trail number of 5-trail 5-fold cross-validation')
    parser.add_argument('--k_value', default=0, type=int, help='fold number of 5-trail 5-fold cross-validation')
    parser.add_argument('--train_class_num', default=2, type=int, help='Classes used in training')
    parser.add_argument('--test_class_num', default=3, type=int, help='Classes used in testing')
    parser.add_argument('--includes_all_train_class', default=True,  action='store_true',
                        help='If required all known classes included in testing')
    parser.add_argument('--method_name', default='plud', type=str, help='Method Name')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate without training')
    parser.add_argument('--save_results', default='True', type=str,
                        help='If save evaluation results to file')
    parser.add_argument('--save_models', default='False', type=str,
                        help='If save evaluation results to file')

    # Parameters for weibull distribution fitting.
    parser.add_argument('--weibull_tail', default=20, type=int, help='Classes used in testing')
    parser.add_argument('--weibull_alpha', default=2, type=int, help='Classes used in testing')
    parser.add_argument('--weibull_threshold', default=0.5, type=float, help='Classes used in testing')
    args = parser.parse_args()
    return args

def main():
    # Parsing args
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading data
    print('==> Preparing data..')
    p_v, k_v = args.p_value, args.k_value
    random_seed = p_v
    batch_size = 32
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    data_dir = os.path.join('../data', args.dataset) 
    kfold_train_idx, kfold_test_idx = get_kfold_img_idx(p=p_v, k=k_v, dataset=args.dataset, data_dir=data_dir, sample_type=None)
    open_kfold_train_idx, old_new_lbl_map_train, old_new_lbl_map_val = get_open_img_idx(args.dataset, kfold_train_idx, args.train_class_num, args.test_class_num, random_seed=random_seed)
    open_kfold_val_idx, _, _ = get_open_img_idx(args.dataset, kfold_train_idx, args.train_class_num, args.test_class_num, random_seed=random_seed, sample_type='random_uniform')
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
    output_params = list(map(id, net.last_linear.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, net.parameters())
    optimizer = optim.SGD([{'params': feature_params},
                        {'params': net.last_linear.parameters(), 'lr': args.lr * 10}],
                        lr=args.lr, weight_decay=0.001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [35, 75], gamma=0.7)

    # Training
    if not args.evaluate:
        train_plain(train_iter, test_iter, net, criterion, optimizer, scheduler, device, args.es)
    
    # Evaluation
    labels, \
    softmax_logits, \
    openmax_logits, \
    softmax_probs, \
    openmax_probs, \
    softmax_preds, \
    softmax_threshold_preds, \
    openmax_preds = test(args, net, train_iter, test_iter, device)

    val_labels, \
    val_softmax_logits, \
    _, \
    _, \
    _, \
    _, \
    _, \
    _ = test(args, net, train_iter, val_iter, device)

    # save results
    data_rename = f'{args.dataset}_tr{args.train_class_num}_te{args.test_class_num}'
    if args.save_results in ['True', 'true', True]:
        curr_dir = os.path.dirname(__file__)
        save_dir_so = os.path.join(curr_dir, '../output/result', data_rename, 'softmax', backbone)
        if not os.path.exists(save_dir_so):
            os.makedirs(save_dir_so)
        write_result_to_file(save_dir_so, softmax_preds, labels, softmax_logits, p=p_v, k=k_v)

        save_dir_val = os.path.join(curr_dir, '../output/result', data_rename, 'softmax_val', backbone)
        if not os.path.exists(save_dir_val):
            os.makedirs(save_dir_val)
        write_result_to_file(save_dir_val, val_labels, val_labels, val_softmax_logits, p=p_v, k=k_v)

        save_dir_op = os.path.join(curr_dir, '../output/result', data_rename, 'openmax', backbone)
        if not os.path.exists(save_dir_op):
            os.makedirs(save_dir_op)
        write_result_to_file(save_dir_op, openmax_preds, labels, openmax_logits, p=p_v, k=k_v)

        save_dir_soth = os.path.join(curr_dir, '../output/result', data_rename, 'softmax_thresh', backbone)
        if not os.path.exists(save_dir_soth):
            os.makedirs(save_dir_soth)
        write_result_to_file(save_dir_soth, softmax_threshold_preds, labels, p=p_v, k=k_v)

    if args.save_models in ['True', 'true']:
        curr_dir = os.path.dirname(__file__)
        model_dir = os.path.join(curr_dir, f'../output/model/{data_rename}/')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = model_dir + f'p{p_v}_k{k_v}_{backbone}_baseline.pth'
        torch.save(net.state_dict(), model_dir)
        print(f'Model saved to {model_dir} successfully!')

'''
This function is modified from https://github.com/ma-xu/Open-Set-Recognition
'''
def test(args, net, trainloader, testloader, device, tailsizes=None):
    net.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            scores.append(outputs)
            labels.append(targets)

    # Get the predict results.
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :] # N x 1 x C
    labels = np.array(labels) # 1 x N

    # Fit the weibull distribution from training data.
    print("Fittting Weibull distribution...")
    _, mavs, dists = compute_train_scores_mavs_dists(args.train_class_num, trainloader, device, net)
    categories = list(range(0, args.train_class_num))
    weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")

    softmax_preds, softmax_threshold_preds, openmax_preds = [], [], []
    softmax_probs, openmax_probs = [], [] 
    softmax_logits, openmax_logits = [], [] 
    for score in scores:
        openmax_prob, softmax_prob, openmax_logit, softmax_logit = openmax(weibull_model, categories, score,
                         0.5, args.weibull_alpha, "euclidean", if_get_logit=True)  # openmax_prob, softmax_prob
        softmax_preds.append(np.argmax(softmax_prob)) # predicted labels
        softmax_threshold_preds.append(np.argmax(softmax_prob) if np.max(softmax_prob) >= args.weibull_threshold else args.train_class_num)
        openmax_preds.append(np.argmax(openmax_prob) if np.max(openmax_prob) >= args.weibull_threshold else args.train_class_num)
        softmax_probs.append(softmax_prob)
        openmax_probs.append(openmax_prob)
        softmax_logits.append(softmax_logit)
        openmax_logits.append(openmax_logit)
        
    return labels, softmax_logits, openmax_logits, softmax_probs, openmax_probs, softmax_preds, softmax_threshold_preds, openmax_preds

if __name__ == '__main__':
    main()

