# coding:utf-8
# This code is based on https://github.com/TingsongYu/PyTorch_Tutorial/blob/master/Code/1_data_prepare/1_3_generate_txt.py.
import os
import argparse

def parse_args():
    # set arg parser
    parser = argparse.ArgumentParser(description='baseline')
    parser.add_argument('--dataset', type=str, default='FLSMDD')
    parser.add_argument('--train_prop', type=float, default=1.0)
    args = parser.parse_args()
    return args

def gen_dir_lbl_list(train_save_path, test_save_path, img_dir, train_prop):
    label_idx = -1
    f_tr = open(train_save_path, 'w')
    f_te = open(test_save_path, 'w')
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  
        sorted_s_dirs = [sub_dir for sub_dir in s_dirs] # sort the folders by folder name
        sorted_s_dirs.sort()
        for sub_dir in sorted_s_dirs:
            label_idx += 1
            i_dir = os.path.join(root, sub_dir)   
            img_list = os.listdir(i_dir)                    
            for i in range(len(img_list)):
                if not (img_list[i].endswith('png') or img_list[i].endswith('jpg')):         
                    continue
                label = label_idx
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + str(label) + '\n'
                if i < int(train_prop*len(img_list)):
                    f_tr.write(line)
                else:
                    f_te.write(line)
    f_tr.close()
    f_te.close()

if __name__ == '__main__':
    args = parse_args()
    assert args.dataset in ['FLSMDD', 'NKSID']
    train_save_path = os.path.join(f'../data/{args.dataset}', 'train_abs.txt')
    test_save_path = os.path.join(f'../data/{args.dataset}', 'test_abs.txt')
    img_dir = os.path.join(f'../data/{args.dataset}')
    gen_dir_lbl_list(train_save_path, test_save_path, img_dir, args.train_prop)
    print(f'The generated file has been saved to {train_save_path}.')
