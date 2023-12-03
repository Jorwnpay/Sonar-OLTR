'''
This file is based on https://github.com/ma-xu/Open-Set-Recognition
'''
import numpy as np
import scipy.spatial.distance as spd
from heapq import nsmallest
import torch
import torch.nn as nn

import libmr


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:] 
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos', if_get_logit=False):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)

    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(input_score.shape[-1])
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    if if_get_logit:
        scores = np.mean(scores, axis=0)
        scores_u = np.mean(np.sum(scores_u, axis=1), axis=0)
        openmax_logit = np.array(scores.tolist() + [scores_u])
        softmax_logit = input_score.ravel()
        return openmax_prob, softmax_prob, openmax_logit, softmax_logit
    return openmax_prob, softmax_prob

def openmax_crosr(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos', if_get_logit=False):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    train_class_num = len(categories)
    score_len = input_score.shape[-1]
    cls_logits = input_score[:,:train_class_num]

    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(score_len)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c in categories:
            mav, dist, model = query_weibull(c, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)
    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(cls_logits.ravel()))

    if if_get_logit:
        scores = np.mean(scores, axis=0)
        scores_u = np.mean(np.sum(scores_u, axis=1), axis=0)
        openmax_logit = np.array(scores.tolist() + [scores_u])
        softmax_logit = cls_logits.ravel()
        return openmax_prob, softmax_prob, openmax_logit, softmax_logit
    return openmax_prob, softmax_prob

def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_train_scores_mavs_dists(train_class_num,trainloader,device,net,if_get_babels_logits=False):
    scores = [[] for _ in range(train_class_num)]
    y_true = []
    logits = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # this must cause error for cifar
            outputs = net(inputs)
            y_true.append(targets)
            logits.append(outputs)
            for score, t in zip(outputs, targets):
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                if torch.argmax(score) == t:    # 保存每个类别分类正确的score
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists = [compute_channel_distances(mav, score) for mav, score in zip(mavs, scores)]
    if if_get_babels_logits:
        y_true = torch.cat(y_true)
        logits = torch.cat(logits)
        print(f'@ y_true.shape is {y_true.shape}')
        print(f'@ logits.shape is {logits.shape}')
        return scores, mavs, dists, y_true, logits
    return scores, mavs, dists

def compute_train_scores_mavs_dists_crosr(train_class_num,trainloader,device,net):
    fusing_feats = [[] for _ in range(train_class_num)]
    with torch.no_grad():
        pool_layer = nn.AdaptiveAvgPool2d((1,1))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # this must cause error for cifar
            outputs, _, z_feats = net(inputs)
            fusing_feat = []
            fusing_feat.append(outputs)
            for z_feat in z_feats:
                pooled_z_feat = torch.squeeze(pool_layer(z_feat), dim=3)
                pooled_z_feat = torch.squeeze(pooled_z_feat, dim=2)
                fusing_feat.append(pooled_z_feat)
            fusing_feat = torch.cat(fusing_feat, dim=1)
            for score, t, fus_feat in zip(outputs, targets, fusing_feat):
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                if torch.argmax(score) == t:
                    fusing_feats[t].append(fus_feat.unsqueeze(dim=0).unsqueeze(dim=0))
    fusing_feats = [torch.cat(x).cpu().numpy() for x in fusing_feats]  # (N_corr, 1, Feat_dim) * C
    mavs = np.array([np.mean(x, axis=0) for x in fusing_feats])  # (C, 1, Feat_dim)
    dists = [compute_channel_distances(mav, fus_feat) for mav, fus_feat in zip(mavs, fusing_feats)]
    return fusing_feats, mavs, dists

def compute_train_recon_loss(train_class_num, trainloader, device, net):
    scores = [[] for _ in range(train_class_num)]
    recon_losses = [[] for _ in range(train_class_num)]
    with torch.no_grad():
        recon_criterion = nn.MSELoss(reduction='none')
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            preds, recon = net(inputs)
            for score, t, re, input in zip(preds, targets, recon, inputs):
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
                    recon_loss = torch.mean(recon_criterion(input, re)).cpu().item()
                    recon_losses[t].append(recon_loss)
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists = [compute_channel_distances(mav, score) for mav, score in zip(mavs, scores)]
    return recon_losses, scores, mavs, dists    

    