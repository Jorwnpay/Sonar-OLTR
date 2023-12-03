# coding: utf-8
"""
 ╦╔═╗╦═╗╦ ╦╔╗╔╔═╗╔═╗╦ ╦
 ║║ ║╠╦╝║║║║║║╠═╝╠═╣╚╦╝
╚╝╚═╝╩╚═╚╩╝╝╚╝╩  ╩ ╩ ╩ 
@time: 2023/04/02
@file: analyse_result.py                
@author: Jorwnpay                    
@contact: jwp@mail.nankai.edu.cn                                         
""" 
import os
import torch
from torch.utils.data import dataset
from my_utils import *
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from scipy import interpolate
import matplotlib.pyplot as plt
from collections import Counter
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Get 5-trail 5-fold results from the saved files. 
    @ dataset : name of dataset, can be KLSG or FLSMDD
    @ backbone : name of backbone, can be resnet18, resnet34, resnet50, vgg16 or vgg19
    @ method : name of method, can be baseline or betl
    @ if_get_logits : if you want to get logits output, default is false
    @ trails : the number of trails results you want to get, default is 5
    @ folds : the number of folds results you want to get, default is 5
'''
def get_y_and_logits_results(dataset, backbone, method='baseline', if_get_logits=False, if_get_close_y_hat=False, trails=5, folds=5):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../output/result', dataset, method, backbone) 
    y_hat = []
    close_y_hat = []
    y_true = []
    logits = []
    for p in range(trails):
        for k in range(folds):
            if (if_get_logits and if_get_close_y_hat):
                y_h, y_t, lo, c_y_h = read_result_from_file(data_dir, p=p, k=k, if_get_logits=if_get_logits, if_get_close_y_hat=if_get_close_y_hat)
                logits = logits + lo
                c_y_h = close_y_hat + c_y_h
            elif if_get_logits:
                y_h, y_t, lo = read_result_from_file(data_dir, p=p, k=k, if_get_logits=if_get_logits)
                logits = logits + lo
            elif if_get_close_y_hat:
                y_h, y_t, c_y_h = read_result_from_file(data_dir, p=p, k=k, if_get_close_y_hat=if_get_close_y_hat)
                c_y_h = close_y_hat + c_y_h
            else:
                y_h, y_t = read_result_from_file(data_dir, p=p, k=k, if_get_logits=if_get_logits)
            y_hat = y_hat + y_h
            y_true = y_true + y_t
    if (if_get_logits and if_get_close_y_hat):
        return y_true, y_hat, logits, close_y_hat
    elif if_get_logits:
        return y_true, y_hat, logits
    elif if_get_close_y_hat:
        return y_true, y_hat, close_y_hat
    return y_true, y_hat

'''
Calculate open-set y_hat while TPR == 95%
    @ y_true : ground truth label of test samples
    @ logits : scores of test samples output from model 
    @ kn_score : scores for detecting known/unknown class, default is max logit
'''
def get_tpr95_y_hat(y_true, logits, kn_score=[]):
    y_hat = np.argmax(logits, axis=1)
    
    unk_lbl = -1 if min(y_true) < 0 else max(y_true)
    if unk_lbl < 0:
        kn_flag = np.array(y_true) > unk_lbl # known flag
    else:
        kn_flag = np.array(y_true) < unk_lbl
    if len(kn_score) > 0:
        max_score = kn_score
    else:
        max_score = np.max(logits, axis=1)
    fpr, tpr, thresholds = roc_curve(kn_flag, max_score) 
    thresh = thresholds[np.abs(np.array(tpr) - 0.95).argmin()]
    y_hat[max_score <= thresh] = unk_lbl
    y_hat = y_hat.tolist()
    return y_hat

'''
Calculate Gmean result of each trail
    @ dataset : name of dataset, can be KLSG or FLSMDD
    @ backbone : name of backbone, can be resnet18, resnet34, resnet50, vgg16 or vgg19
    @ method : name of method, can be baseline or betl
    @ trails : the number of trails results you want to get, default is 5
    @ folds : the number of folds results you want to get, default is 5
'''
def get_gmean_each_trail(dataset, backbone, method='baseline', trails=5, folds=5):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../output/result', dataset, method, backbone)
    gmean = [] 
    for p in range(trails):
        y_hat = []
        y_true = []
        for k in range(folds):
            y_h, y_t = read_result_from_file(data_dir, p=p, k=k)
            y_hat = y_hat + y_h
            y_true = y_true + y_t
        gmean.append(cal_gmean(y_true, y_hat))
    return gmean    

# @ thresh_mode: if thresh_mode == None, then use default y_hat; else use tpr95
def get_osfm_each_trail(dataset, backbone, method='baseline', trails=5, folds=5, thresh_mode=None):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../output/result', dataset, method, backbone)
    osfm_list = [] 
    mac_f1_list = []
    for p in range(trails):
        y_hat = []
        y_true = []
        logits = []
        for k in range(folds):
            y_h, y_t, lo = read_result_from_file(data_dir, p=p, k=k, if_get_logits=True)
            if thresh_mode == 'tpr95' and 'cssr' not in method:
                y_h = get_tpr95_y_hat(y_t, lo)
            if 'mdl4ow' in method:
                y_h, y_t, lo, ws = read_result_from_file(data_dir, p=p, k=k, if_get_logits=True, if_get_close_y_hat=True)
                ws = -1 * np.array(ws)
                y_h = get_tpr95_y_hat(y_t, lo, kn_score=ws)
            y_true = y_true + y_t
            logits = logits + lo
            y_hat = y_hat + y_h
        try:
            osfm_list.append(cal_osfm(y_true, y_hat))
        except:
            print(f'OSFM is calculated failed! Maybe there are unpredicted classes.')
        mac_f1_list.append(f1_score(y_true, y_hat, average='macro'))
    return osfm_list, mac_f1_list

def get_close_eval_each_trail(dataset, backbone, train_class_num, method='baseline', trails=5, folds=5):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../output/result', dataset, method, backbone)
    mac_f1_list = []
    gmean_list = []
    for p in range(trails):
        y_hat = []
        y_true = []
        for k in range(folds):
            y_h, y_t, lo = read_result_from_file(data_dir, p=p, k=k, if_get_logits=True)
            if 'cssr' in method:        # Because the closed set y_h of cssr is separately preserved, it is considered separately.
                _, y_t, lo, y_h = read_result_from_file(data_dir, p=p, k=k, if_get_logits=True, if_get_close_y_hat=True)
            else:
                if np.array(lo).shape[-1] > train_class_num:   # if unknown score in logits（openmax、crosr）
                    close_lo = np.array(lo)[:, :-1]            # select logits of closed-set
                else:
                    close_lo = np.array(lo)
                y_h = np.argmax(close_lo, axis=1)
            if not isinstance(y_h, list):
                y_h = y_h.tolist()
            y_true = y_true + y_t
            y_hat = y_hat + y_h
        if min(y_true) < 0: 
            unk_lbl = -1 
        else: 
            unk_lbl = train_class_num
        kn_y_flags = np.array(y_true)!=unk_lbl
        kn_y_true = np.array(y_true)[kn_y_flags]
        kn_y_hat = np.array(y_hat)[kn_y_flags]
        mac_f1_list.append(f1_score(kn_y_true, kn_y_hat, average='macro'))
        gmean_list.append(cal_gmean(kn_y_true, kn_y_hat))
    return mac_f1_list, gmean_list

'''
Draw confusion matrix
    @ class_list : class name list, e.g., ['Plane', 'Wreck']
    @ y_true : the ground truth labels
    @ y_hat : the predict labels
    @ tail_idxes : indexes of tail classes, e.g., [0]
    @ pic_dir : confusion matrix picture save direction
'''
def draw_conf_matrix(class_list, tail_idxes, pic_dir, y_true=[], y_hat=[], confus_matrix=[], missed_classes=[]):
    if len(confus_matrix):
        prob_matrix = np.around(confus_matrix, 3)
    elif len(y_true) and len(y_hat):
        conf_matrix = confusion_matrix(y_true, y_hat)
        print(conf_matrix)
        prob_matrix = np.around((conf_matrix.T/np.sum(conf_matrix, 1)).T, 3)
    else:
        print('ERROR! You should not set both y_true, y_hat and confus_matrix as None')
    print('--------- Drawing confusion matrix ... ----------')
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(prob_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0., vmax=1.)
    plt.colorbar()
    if len(missed_classes):
        used_class_list = []
        used_classes = []
        for i in range(len(class_list)):
            if i not in missed_classes:
                used_class_list.append(class_list[i])
                used_classes.append(i)
        tick_marks = np.arange(len(used_class_list))
        plt.xticks(tick_marks, used_class_list, rotation=45, horizontalalignment='right', family='Times New Roman', fontsize=25)
        plt.yticks(tick_marks, used_class_list, family='Times New Roman', fontsize=25)
        for idx in range(len(used_classes)):
            if used_classes[idx] in tail_idxes:
                plt.gca().get_xticklabels()[idx].set_color('red')
                plt.gca().get_yticklabels()[idx].set_color('red')
    else:
        tick_marks = np.arange(len(class_list))
        plt.xticks(tick_marks, class_list, rotation=45, horizontalalignment='right', family='Times New Roman', fontsize=25)
        plt.yticks(tick_marks, class_list, family='Times New Roman', fontsize=25)
        for idx in tail_idxes:
            plt.gca().get_xticklabels()[idx].set_color('red')
            plt.gca().get_yticklabels()[idx].set_color('red')
    for i in range(len(prob_matrix)):
        for j in range(len(prob_matrix)):
            if prob_matrix[i, j] < 0.6 and j == i:
                plt.annotate(prob_matrix[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center', family='Times New Roman', color='black', fontsize=20, fontweight='bold')
                # plt.annotate(prob_matrix[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center', family='Times New Roman', color='white', fontsize=20, fontweight='bold')
            elif j == i:
                plt.annotate(prob_matrix[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center', family='Times New Roman', color='white', fontsize=20, fontweight='bold')
            else:
                plt.annotate(prob_matrix[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center', family='Times New Roman', fontsize=17)
    plt.tight_layout()
    plt.ylabel('True label', family='Times New Roman', fontsize=30, fontweight='bold')
    plt.xlabel('Predicted label', family='Times New Roman', fontsize=30, fontweight='bold')
    fig.savefig(pic_dir, format='pdf', bbox_inches='tight')
    print(f'Confusion matrix has been saved to {pic_dir}.')

# modified from CSSR
def compute_oscr_macro_cssr(preds, labels, scores, if_get_CCR=False):
    preds = np.array(preds)
    labels = np.array(labels)
    scores = np.array(scores).flatten()
    if min(labels) < 0:                                         # if -1 in labels, means unk_label == -1
        unk_scores = scores[labels < 0]                         # unknown score
        kn_cond = labels >= 0                                   # flag of known samples
    else:                                                       # condition of unk_label == train_class_num
        unk_scores = scores[labels == max(labels)]              # unknown score
        kn_cond = labels < max(labels)                          # flag of known samples
    kn_ct = kn_cond.sum()                                       # number of known samples
    unk_ct = scores.shape[0] - kn_ct                            # number of unknown samples
    kn_scores = scores[kn_cond]                                 # known score
    kn_correct_pred = preds[kn_cond] == labels[kn_cond]         # flag of correct classified known samples 
        
    def fpr(thr):                                               # fpr：Proportion of unknown samples misclassified as known, out of total unknowns.
        return (unk_scores > thr).sum() / unk_ct

    def ccr_micro(thr):                                         # ccr_micro：Proportion of known samples correctly classified, out of total knowns.
        ac_cond = (kn_scores > thr) & (kn_correct_pred)
        return ac_cond.sum() / kn_ct

    def ccr_macro(thr, labels):                                 # ccr_macro：Avg. proportion of correctly classified for each known class, out of class totals.
        CCs = []
        for cls in set(labels):
            if (min(labels) < 0 and cls < 0) or (min(labels) >= 0 and cls >= max(labels)):   # exclude unknown class
                continue
            flag = labels==cls                                  # Obtain sample labels of the cls class from the test set tags.
            cls_ct = flag.sum()                                 # Get the number of samples with true labels as the cls class.
            cls_scores = scores[flag]                           # Select the scores of samples with true labels as the cls class.
            cls_correct_pred = preds[flag]==labels[flag]        # Acquire the correct prediction marks for samples with true labels as the cls class.
            ac_cond = (cls_scores > thr) & (cls_correct_pred)   # Obtain marks for samples with true labels as the cls class where the scores exceed a threshold and predictions are correct.
            CCs.append(ac_cond.sum() / cls_ct)                  # Calculate CCR for samples with true labels as the cls class.
        CCs_Mac = np.array(CCs).mean()
        CCs_Gmean = np.power(np.prod(CCs), 1/len(CCs))
        return CCs_Mac, CCs_Gmean

    sorted_scores = -np.sort(-scores)                           # Sort the scoring results in descending order.
    # Cutoffs are of prediction values
        
    CCR_Micro = [0]
    CCR_Macro = [0]
    CCR_Gmean = [0]
    FPR = [0] 

    for s in sorted_scores:
        CCR_Micro.append(ccr_micro(s))
        CCR_Macro.append(ccr_macro(s, labels)[0])
        CCR_Gmean.append(ccr_macro(s, labels)[1])
        FPR.append(fpr(s))
    CCR_Micro += [1]
    CCR_Macro += [1]
    CCR_Gmean += [1]
    FPR += [1]

    # Positions of ROC curve (FPR, TPR)
    ROC_Micro = sorted(zip(FPR, CCR_Micro), reverse=True)
    ROC_Macro = sorted(zip(FPR, CCR_Macro), reverse=True)
    ROC_Gmean = sorted(zip(FPR, CCR_Gmean), reverse=True)
    OSCR_Micro, OSCR_Macro, OSCR_Gmean = 0, 0, 0
    # Compute AUROC Using Trapezoidal Rule
    for j in range(len(CCR_Micro)-1):
        h = ROC_Micro[j][0] - ROC_Micro[j+1][0]
        w = (ROC_Micro[j][1] + ROC_Micro[j+1][1]) / 2.0
        OSCR_Micro = OSCR_Micro + h*w
        h = ROC_Macro[j][0] - ROC_Macro[j+1][0]
        w = (ROC_Macro[j][1] + ROC_Macro[j+1][1]) / 2.0
        OSCR_Macro = OSCR_Macro + h*w
        h = ROC_Gmean[j][0] - ROC_Gmean[j+1][0]
        w = (ROC_Gmean[j][1] + ROC_Gmean[j+1][1]) / 2.0
        OSCR_Gmean = OSCR_Gmean + h*w
    if if_get_CCR:
        return CCR_Macro, FPR, (OSCR_Micro, OSCR_Macro, OSCR_Gmean)
    return OSCR_Micro, OSCR_Macro, OSCR_Gmean

def compute_oscr_macro(known_logits, known_labels, unk_logits, num_class, save_dir=None):
    max_known_logit, max_unk_logit = np.max(known_logits, axis=1), np.max(unk_logits, axis=1)   # 已知类、未知类得分的最大值       
    known_pred = np.argmax(known_logits, axis=1)                                                # 已知类的预测标签
    correct = (known_pred == known_labels)          
    known_corr_flag = np.zeros(len(max_known_logit))        
    known_corr_flag[known_pred == known_labels] = 1    
    k_TP_flag = np.concatenate((known_corr_flag, np.zeros(len(max_unk_logit))), axis=0)
    u_FP_flag = np.concatenate((np.zeros(len(max_known_logit)), np.ones(len(max_unk_logit))), axis=0)
    max_logit = np.concatenate((max_known_logit, max_unk_logit), axis=0)
    n = len(max_logit)
    labels = np.concatenate((np.array(known_labels), num_class*np.ones(len(max_unk_logit))), axis=0)
    
    CCR = [0 for x in range(n+2)]
    FPR = [0 for x in range(n+2)] 
    idx = max_logit.argsort()
    sorted_k_TP_flag = k_TP_flag[idx]
    sorted_u_FP_flag = u_FP_flag[idx]
    sorted_labels = labels[idx]
    sample_num_per_cls = Counter(labels)

    for k in range(n-1):   
        CCs = []
        for cls in set(known_labels):
            cls_target = [sorted_k_TP_flag[i] if sorted_labels[i] == cls else 0 for i in range(k+1, len(sorted_k_TP_flag))]
            CC_cls = float(np.array(cls_target).sum()) / float(sample_num_per_cls[cls])
            CCs.append(CC_cls)
        CC = np.array(CCs).mean()
        FP = sorted_u_FP_flag[k:].sum()
        # True	Positive Rate
        CCR[k] = float(CC)
        # False Positive Rate
        FPR[k] = float(FP) / float(len(max_unk_logit))

    CCR[n] = 0.0                         
    FPR[n] = 0.0                        
    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)
    OSCR = 0
    # Compute AUROC Using Trapezoidal Rule
    for j in range(n+1):                 
        h =   ROC[j][0] - ROC[j+1][0]
        w =  (ROC[j][1] + ROC[j+1][1]) / 2.0
        OSCR = OSCR + h*w

    if save_dir:
        # setup plot details    
        fig, ax = plt.subplots(figsize=(10, 8))
        display = RocCurveDisplay(
                fpr=FPR,
                tpr=CCR,
                roc_auc=OSCR,
            )
        display.plot(ax=ax, name=f"oscr_macro", color="red")
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('TPR', fontsize=30, fontweight='bold')
        ax.set_ylabel('OSCR-Macro', fontsize=30, fontweight='bold')
        fig.savefig(save_dir, format='pdf', bbox_inches='tight')

    return OSCR

def get_oscr_macro_each_trail(dataset, backbone, num_class, train_class_num, method='baseline', trails=5, folds=5, thresh_mode=None, if_get_CCR=False):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../output/result', dataset, method, backbone)
    oscr_mac_list = [] 
    ccr_mac_list = []
    fpr_mac_list = []
    for p in range(trails): 
        y_hat, y_true, logits = [], [], [] 
        for k in range(folds):
            y_h, y_t, lo = read_result_from_file(data_dir, p=p, k=k, if_get_logits=True)
            if thresh_mode == 'tpr95':      # 将已知类里的95%正确分为已知类的阈值
                if 'cssr' in method:        # 因为cssr默认就是tpr95得到的y_h，故单独考虑
                    y_h = y_h
                elif 'mdl4ow' in method:    # mdl4ow的已知/未知类判别得分不一样，故单独考虑
                    _, y_t, lo, ws = read_result_from_file(data_dir, p=p, k=k, if_get_logits=True, if_get_close_y_hat=True)
                    ws = -1 * np.array(ws)
                    y_h = get_tpr95_y_hat(y_t, lo, kn_score=ws)
                    lo = ws
                else:
                    y_h = get_tpr95_y_hat(y_t, lo)
            elif thresh_mode == 'close_pred':   # 使用闭集预测结果
                if 'cssr' in method:        # 因为cssr的闭集y_h单独保存了下来，故单独考虑
                    _, y_t, lo, y_h = read_result_from_file(data_dir, p=p, k=k, if_get_logits=True, if_get_close_y_hat=True)
                elif 'mdl4ow' in method:    # mdl4ow的已知/未知类判别得分不一样，故单独考虑
                    _, y_t, lo, ws = read_result_from_file(data_dir, p=p, k=k, if_get_logits=True, if_get_close_y_hat=True)
                    ws = -1 * np.array(ws)
                    y_h = np.argmax(lo, axis=1)
                    lo = ws
                else:
                    if np.array(lo).shape[-1] > train_class_num:   # logits中有未知类（openmax、crosr）
                        close_lo = np.array(lo)[:, :-1]            # 取出闭集logit
                    else:
                        close_lo = np.array(lo)
                    y_h = np.argmax(close_lo, axis=1)
                    lo = close_lo
            if len(np.array(lo).shape) > 1 and np.array(lo).shape[-1] > 1: # logits不是只有最大logit值，而是原始logits的情况，即[[1,2,3], [1,2,3], ...]
                lo = np.max(lo, axis=1)
            if if_get_CCR:
                ccr_mac, fpr_mac, oscr_mac = compute_oscr_macro_cssr(y_h, y_t, lo, if_get_CCR)
                ccr_mac_list.append(ccr_mac)
                fpr_mac_list.append(fpr_mac)
                oscr_mac_list.append(oscr_mac)
            else:
                oscr_mac_list.append(compute_oscr_macro_cssr(y_h, y_t, lo))
    if if_get_CCR:
        return oscr_mac_list, ccr_mac_list, fpr_mac_list
    else:
        return oscr_mac_list

def draw_macro_oscr_curve(dataset, backbone, num_class, train_class_num, save_dir):
    # setup plot details    
    fig, ax = plt.subplots(figsize=(10, 8))
    ltsid_method_list = ['softmax', 'openmax_100e', 'crosr_100e', 'mdl4ow_100e', 'proser_dum_1_2.0', 'proser_dum_5_2.0', 'cssr_100e', '2_stage_softmax_100e', 'marc_softmax_100e', 'betl_softmax', 'oltr_100e', 'plud_new_feat_mixup_2_gamma_0.5']
    flsmdd_method_list = ['softmax_100e', 'openmax_100e', 'crosr_100e', 'mdl4ow_100e', 'proser_dum_1_2.0', 'proser_dum_5_2.0', 'cssr_100e', '2_stage_softmax_100e', 'marc_softmax_100e', 'betl_softmax_100e', 'oltr_100e', 'plud_final_gamma_0.5']
    nksid_method_list = ['softmax_100e', 'openmax_100e', 'crosr_100e', 'mdl4ow_100e', 'proser_dum_1', 'proser_dum_5', 'cssr_100e', '2_stage_softmax_100e', 'marc_softmax_100e', 'betl_softmax_100e', 'oltr_100e', 'plud_final_gamma_0.5']
    method_dict = {
        'LTSID_tr2_te8': nksid_method_list,
        'LTSID_tr3_te8': nksid_method_list,
        'LTSID_tr7_te8': nksid_method_list,
        'LTSID_tr5_te8': ltsid_method_list,
        'FLSMDD_tr2_te10': nksid_method_list,
        'FLSMDD_tr4_te10': nksid_method_list,
        'FLSMDD_tr8_te10': nksid_method_list,
        'FLSMDD_tr6_te10': flsmdd_method_list,
        'NKSID_tr2_te8': nksid_method_list,
        'NKSID_tr3_te8': nksid_method_list,
        'NKSID_tr5_te8': nksid_method_list,
        'NKSID_tr7_te8': nksid_method_list,
    }
    method_list = method_dict[dataset]
    colors = ['black', 'dimgray', 'tomato', 'silver', 'sandybrown', 'gold', 'dodgerblue', 'turquoise', 'seagreen', 'violet', 'purple', 'red']
    show_method_list = ['SoftMax', 'OpenMax', 'CROSR', 'MDL4OW', 'Proser(DN=1)', 'Proser(DN=5)', 'CSSR', 'CE-DRS', 'MARC', 'BETL', 'OLTR', 'PLUD(Ours)']
    line_styles = ['--', '--', '--', '--', '--', '--', '--', '-.', '-.', '-.', '-', '-']

    for idx, method in enumerate(method_list):
        # calculate macro average CCR_macro and TPR of all folds
        oscr_lists, ccr_list, fpr_list = get_oscr_macro_each_trail(dataset, backbone, num_class, train_class_num, method, thresh_mode='close_pred', if_get_CCR=True)
        _, avg_oscr_macro, _ = np.mean(oscr_lists, axis=0)
        print(avg_oscr_macro)
        inter_func_dict = dict()
        for i in range(len(fpr_list)):
            inter_func_dict[i] = interpolate.interp1d(fpr_list[i], ccr_list[i], kind='linear')
        all_fpr = np.unique(np.concatenate(fpr_list))
        mean_ccr = np.zeros_like(all_fpr)
        for i in range(len(fpr_list)):
            mean_ccr += inter_func_dict[i](all_fpr)
        mean_ccr /= len(fpr_list)

        # draw macro average TPR-CCR-macro curves of all classes
        display = RocCurveDisplay(
            fpr=all_fpr,
            tpr=mean_ccr,
            roc_auc=avg_oscr_macro,
        )
        display.plot(ax=ax, linestyle=line_styles[idx], name=f'{show_method_list[idx]}', color=colors[idx])

    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('FPR', fontsize=30, fontweight='bold')
    ax.set_ylabel('CCR-macro', fontsize=30, fontweight='bold')
    fig.savefig(save_dir, format='pdf', bbox_inches='tight')
    print(f'Macro average OSCR-macro curves of all folds have been saved to {save_dir}.')
        
'''
Calculate Normalized Accuracy (NA)
'''
def cal_norm_acc_one_trail(y_true, y_hat, lamda=0.5):
    cls_list = set(y_true)
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    if min(cls_list) == -1:      
        unk_cls = -1
    else:
        unk_cls = max(cls_list)   
    unk_flag = y_true==unk_cls
    unk_y_hat = y_hat[unk_flag]
    unk_y_true = y_true[unk_flag]
    kn_flag = y_true!=unk_cls
    kn_y_hat = y_hat[kn_flag]
    kn_y_true = y_true[kn_flag]
    
    unk_corr = unk_y_hat==unk_y_true
    aus = unk_corr.sum() / len(unk_corr)    
    kn_corr = kn_y_hat==kn_y_true           
    aks = kn_corr.sum() / len(kn_corr)    
    na = lamda * aks + (1 - lamda) * aus  
    return na

'''
Calculate Normalized Macro Accuracy (NMA)
'''
def cal_norm_mac_acc_one_trail(y_true, y_hat, lamda=0.5):
    cls_list = set(y_true)
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    unk_y_hat_list, unk_y_true_list = [], []
    kn_y_hat_list, kn_y_true_list = [], []
    if min(cls_list) == -1:      
        unk_cls = -1
    else:
        unk_cls = max(cls_list)   
    for cls in cls_list:
        flag = y_true==cls
        if cls == unk_cls:
            unk_y_hat_list = y_hat[flag]
            unk_y_true_list = y_true[flag]
        else:
            kn_y_hat_list.append(y_hat[flag])
            kn_y_true_list.append(y_true[flag])
    unk_corr = unk_y_hat_list==unk_y_true_list
    aus = unk_corr.sum() / len(unk_corr)       
    aks_mac = 0
    for kn_y_h, kn_y_t in zip(kn_y_hat_list, kn_y_true_list):
        kn_corr = kn_y_h==kn_y_t
        aks_mac += kn_corr.sum() / len(kn_corr)
    aks_mac = aks_mac / len(kn_y_hat_list)     
    nma = lamda * aks_mac + (1 - lamda) * aus   
    return nma

def cal_norm_mac_acc(dataset, backbone, method='baseline', trails=5, folds=5, thresh_mode=None):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../output/result', dataset, method, backbone)
    nma_list = [] 
    na_list = []
    for p in range(trails):
        y_hat = []
        y_true = []
        for k in range(folds):
            y_h, y_t, lo = read_result_from_file(data_dir, p=p, k=k, if_get_logits=True)
            if thresh_mode == 'tpr95' and 'cssr' not in method:
                y_h = get_tpr95_y_hat(y_t, lo)
            if 'mdl4ow' in method:
                y_h, y_t, lo, ws = read_result_from_file(data_dir, p=p, k=k, if_get_logits=True, if_get_close_y_hat=True)
                ws = -1 * np.array(ws)
                y_h = get_tpr95_y_hat(y_t, lo, kn_score=ws)
            y_hat = y_hat + y_h
            y_true = y_true + y_t
        nma_list.append(cal_norm_mac_acc_one_trail(y_true, y_hat))
        na_list.append(cal_norm_acc_one_trail(y_true, y_hat))
    return nma_list, na_list

'''
Write results to file 
    @ res : results
    @ res_dir: results saving direction
    @ res_name: result's name, e.g., Macro-F1...
    @ dataset : name of dataset, can be LTSID, FLSMDD or NKSID
    @ method : name of method, can be baseline or betl
    @ backbone : name of backbone, can be resnet18, resnet34, resnet50, vgg16 or vgg19
'''
def write_results(res, res_dir, res_name, dataset, method, backbone):
    print(f'--------- Writing {res_name} results ... ----------')
    file = open(res_dir, 'a')
    file.write(f'[Time: {time.asctime()}] {res_name} of {dataset}_{method}_{backbone} is : {res*100:.2f}\n')
    print(f'{res_name} result: {res} have been saved to {res_dir}.')

def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='analyse result')
    parser.add_argument('--dataset', type=str, default='NKSID')
    parser.add_argument('--method', type=str, default='plud')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--thresh_mode', type=str, default='tpr95', help='Choose a threshold mode for calculating y_hat from logit, default is tpr95')
    parser.add_argument('--train_class_num', default=5, type=int, help='Classes used in training')
    parser.add_argument('--test_class_num', default=8, type=int, help='Classes used in testing')
    parser.add_argument('--get_osfm', type=str, default='True', help='If you want to get Macro-F1 and OSFM result, default is True.')
    parser.add_argument('--get_oscr', type=str, default='True', help='If you want to get OSCR result, default is True.')
    parser.add_argument('--get_oscr_curve', type=str, default='False', help='If you want to get OSCR curves result, default is False.')
    parser.add_argument('--get_nma', type=str, default='True', help='If you want to get NMA result, default is True.')
    parser.add_argument('--get_close_eval', type=str, default='False', help='If you want to get close-set Gmean and Macro-F1 result, default is False.')
    parser.add_argument('--show_pic', type=str, default='False', help='If you want to show confusion matrix or Precision-Recall curves, default is False.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # get args
    args = parse_args()

    # set params
    thresh_mode = args.thresh_mode
    dataset = args.dataset
    dataset_rename = f'{dataset}_tr{args.train_class_num}_te{args.test_class_num}'
    method = args.method
    backbone = args.backbone
    plt.rc('font', family='Times New Roman', size=19) # size=19
    curr_dir = os.path.dirname(__file__)
    
    if dataset == 'FLSMDD':
        num_class = 10
        class_list = ['Bottle', 'Can', 'Chain', 'D_Carton', 'Hook',
                    'Propeller', 'Sh_Bottle', 'St_Bottle', 'Tire', 'Valve', 'Unknown']
        tail_idxes = [2, 4, 5, 6, 7, 9]
        colors = ['lightseagreen', 'steelblue', 'deeppink', 'blue', 'orangered', 
                  'lightpink', 'coral', 'magenta', 'cyan', 'red']
    elif dataset == 'NKSID':
        num_class = 8
        class_list = ['B_Propeller', 'Cylinder', 'F_Net', 'Floats', 
                    'I_Pipeline', 'S_Propeller', 'S_Pipeline', 'Tire', 'Unknown']
        tail_idxes = [0, 2, 4, 5, 6]
        colors = ['orangered', 'lightseagreen', 'lightpink', 'steelblue', 
                  'coral', 'deeppink', 'magenta', 'cyan']
    else:
        print(f'ERROR! DATASET {dataset} IS NOT EXIST!')

    # set saving directory
    save_dir = os.path.join(curr_dir, '../output/display')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    # write osfm
    if args.get_osfm in ['True', 'true']:
        osfm_dir = os.path.join(save_dir, 'osfm.txt')
        mac_f1_dir = os.path.join(save_dir, 'mac_f1.txt')
        osfm_list, macro_f1_list = get_osfm_each_trail(dataset_rename, backbone, method, thresh_mode=thresh_mode)
        avg_osfm = np.mean(osfm_list)
        avg_mac_f1 = np.mean(macro_f1_list)
        write_results(avg_osfm, osfm_dir, 'OSFM', dataset_rename, method, backbone)
        write_results(avg_mac_f1, mac_f1_dir, 'Macro-F1', dataset_rename, method, backbone)

    if args.get_oscr in ['True', 'true']:
        oscr_lists = get_oscr_macro_each_trail(dataset_rename, backbone, num_class, args.train_class_num, method, thresh_mode='close_pred')
        avg_oscr_micro, avg_oscr_macro, avg_oscr_gmean = np.mean(oscr_lists, axis=0)
        oscr_macro_dir = os.path.join(save_dir, 'oscr_macro.txt')
        oscr_micro_dir = os.path.join(save_dir, 'oscr_micro.txt')
        write_results(avg_oscr_macro, oscr_macro_dir, 'OSCR-Mac', dataset_rename, method, backbone)
        write_results(avg_oscr_micro, oscr_micro_dir, 'OSCR-Mic', dataset_rename, method, backbone)

    if args.get_oscr_curve in ['True', 'true']:
        oscr_mac_dir = os.path.join(save_dir, 'oscrmac')
        if not os.path.exists(oscr_mac_dir):
            os.makedirs(oscr_mac_dir) 
        oscr_curve_dir = os.path.join(oscr_mac_dir, '{dataset_rename}.pdf')
        draw_macro_oscr_curve(dataset_rename, backbone, num_class, args.train_class_num, oscr_curve_dir)

    if args.get_nma in ['True', 'true']:
        nma_dir = os.path.join(save_dir, f'nma.txt')
        na_dir = os.path.join(save_dir, f'na.txt')
        nma_list, na_list = cal_norm_mac_acc(dataset_rename, backbone, method, thresh_mode=thresh_mode)
        nma_avg = np.mean(nma_list)
        na_avg = np.mean(na_list)
        write_results(nma_avg, nma_dir, 'NMA', dataset_rename, method, backbone)
        write_results(na_avg, na_dir, 'NA', dataset_rename, method, backbone)

    if args.get_close_eval in ['True', 'true']:
        cf1_dir = os.path.join(save_dir, f'close_f1.txt')
        cgmean_dir = os.path.join(save_dir, f'close_gmean.txt')
        close_f1_list, close_gmean_list = get_close_eval_each_trail(dataset_rename, backbone, args.train_class_num, method)
        close_f1_avg = np.mean(close_f1_list)
        close_gmean_avg = np.mean(close_gmean_list)
        write_results(close_f1_avg, cf1_dir, 'Close-F1', dataset_rename, method, backbone)
        write_results(close_gmean_avg, cgmean_dir, 'Close-Gmean', dataset_rename, method, backbone)

    if args.show_pic in ['True', 'true']:
        plt.show()