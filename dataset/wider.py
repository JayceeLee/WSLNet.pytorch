import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

def img_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            # imlist.append( (impath, int(imlabel)) )
            imlist.append( impath )				
    return imlist

def label_reader(flist):
    """
    flist format: multi-hot-label\n ...
    label example: 0 0 0 1 0 1 ...
    """
    label_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            label = line.strip().split()
            label = torch.Tensor([int(item) for item in label])
            label_list.append( label )					
    return label_list

class WiderClassification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, 
            img_reader=img_reader, label_reader=label_reader, loader=default_loader):
        self.root = root
        img_list = os.path.join(root, 'wider_att_'+set+'_imglist.txt')
        label_list = os.path.join(root, 'wider_att_'+set+'_label.txt')
        self.imlist = img_reader(img_list)
        self.labels = label_reader(label_list)		
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        target = self.labels[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
              
        return img, target

    def __len__(self):
        return len(self.imlist)

    def get_number_classes(self):
        return 14