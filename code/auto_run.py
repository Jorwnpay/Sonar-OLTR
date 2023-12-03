# coding: utf-8
"""
 ╦╔═╗╦═╗╦ ╦╔╗╔╔═╗╔═╗╦ ╦
 ║║ ║╠╦╝║║║║║║╠═╝╠═╣╚╦╝
╚╝╚═╝╩╚═╚╩╝╝╚╝╩  ╩ ╩ ╩ 
@time: 2023/04/02
@file: auto_run.py                
@author: Jorwnpay                    
@contact: jwp@mail.nankai.edu.cn                                         
""" 

import subprocess
import argparse

# Parameters setting
backbone = 'resnet18'
save_results = 'True'
epoches = 100
weibull_tail = 20
weibull_tail_prop = 0.2
weibull_threshold = 0.5
trails = 5
folds = 5
datasets = [
            'FLSMDD',
            'NKSID'
            ]
# Contents: [train_class_num, test_class_num, weibull_alpha]
ntr_nte_alpha_dict = {'FLSMDD': [#[2, 10, 2],
                                #  [4, 10, 3],
                                 [6, 10, 4],
                                #  [8, 10, 5]
                                 ],
                      'NKSID': [#[2, 8, 2],
                                # [3, 8, 2],
                                [5, 8, 3],
                                # [7, 8, 4]
                                ],}

methods = [
            "softmax",
            "plud"
           ]

def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='Auto run script.')
    parser.add_argument('--code_dir', type=str, help='code direction')
    parser.add_argument('--mode', type=str, help='run mode, can be train or analyse')
    parser.add_argument('--method_name', default='plud', type=str, help='Method Name')
    args = parser.parse_args()
    return args

def train(args):
    for dataset in datasets:
        for ntr_nte_alpha in ntr_nte_alpha_dict[dataset]:
            for p in range(trails):
                for k in range(folds):
                    print(f'------- P{p}-K{k}: Dataset is {dataset}, backbone is {backbone}. -------')
                    print(f'@ train_class_num is {ntr_nte_alpha[0]}')
                    print(f'@ test_class_num is {ntr_nte_alpha[1]}')
                    subprocess.run(['python', args.code_dir, 
                                    '--dataset', dataset, 
                                    '--backbone', backbone,
                                    '--es', str(epoches),
                                    '--p_value', str(p),
                                    '--k_value', str(k),
                                    '--train_class_num', str(ntr_nte_alpha[0]),
                                    '--test_class_num', str(ntr_nte_alpha[1]),
                                    '--save_results', save_results,
                                    '--weibull_tail', str(weibull_tail),
                                    '--weibull_alpha', str(ntr_nte_alpha[2]),
                                    '--weibull_threshold', str(weibull_threshold)
                                    ])

def analyse(args):
    for dataset in datasets:
        for ntr_nte_alpha in ntr_nte_alpha_dict[dataset]:
            for method in methods:
                print(f'------- Dataset is {dataset}, method is {method}. -------')
                print(f'@ train_class_num is {ntr_nte_alpha[0]}')
                print(f'@ test_class_num is {ntr_nte_alpha[1]}')
                subprocess.run(['python', args.code_dir, 
                                '--dataset', dataset, 
                                '--method', method,
                                '--thresh_mode', 'tpr95',
                                '--train_class_num', str(ntr_nte_alpha[0]),
                                '--test_class_num', str(ntr_nte_alpha[1]),
                                '--get_osfm', 'True',
                                '--get_oscr', 'True',
                                '--get_conf_matrix', 'False',
                                '--get_nma', 'True'
                                ])
        
def main():
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'analyse':
        analyse(args)
    else:
        print('ERROR! Please input an exist mode: train or analyse.')

if __name__ == '__main__':
    main()