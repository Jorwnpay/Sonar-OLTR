# coding: utf-8
import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    '''
    txt_path: a file which saves paths and labels of images
    img_idx: the index list of images you want, like [1,2,3,8,9,10...]  
    '''
    def __init__(self, txt_path, img_idx, transform=None, target_transform=None, old_new_lbl_map=None, unk_lbl=None):
        fh = open(txt_path, 'r')
        lines = fh.readlines()
        slices = [lines[i] for i in img_idx]
        slices = [slices[i].rstrip().split() for i in range(len(slices))]
        imgs = [(slices[i][0], int(slices[i][1])) for i in range(len(slices))]
        self.imgs = imgs  
        self.transform = transform
        self.target_transform = target_transform
        self.old_new_lbl_map = old_new_lbl_map
        self.unk_lbl = unk_lbl

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        if os.path.isabs(fn): 
            img = Image.open(fn).convert('RGB')
        else:
            curr_dir = os.path.dirname(__file__)
            img = Image.open(os.path.join(curr_dir, fn)).convert('RGB')    
        if self.transform is not None:
            img = self.transform(img) 
        if self.old_new_lbl_map is not None:
            label = self.old_new_lbl_map[label]
        return img, label

    def __len__(self):
        return len(self.imgs)
    