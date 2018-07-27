import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 

class Wish (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, root, split, transform=None, **kargs):
        name2dir = {
                        'train': ['/opt/intern/users/haozhang/wish/dataset/train_list.txt', '/opt/intern/users/haozhang/wish/dataset/validation_list.txt'],
                        'val': ['/opt/intern/users/haozhang/wish/dataset/validation_list.txt'],
                        'test': ['/opt/intern/users/haozhang/wish/dataset/test_list.txt']
                    }
        src_dir = {
                        '/opt/intern/users/haozhang/wish/dataset/train_list.txt': '/home/haozhang/wish/train/',
                        '/opt/intern/users/haozhang/wish/dataset/validation_list.txt': '/home/haozhang/wish/validation/',
                        '/opt/intern/users/haozhang/wish/dataset/test_list.txt': '/home/haozhang/wish/test/',
                  }
        self.img_path = []
        self.label = []
        self.transform = transform
    

        load_file = name2dir[split]
        for imglist in load_file:
            dir = src_dir[imglist]
            with open(imglist) as f:
                for t in f:
                    if split != 'test':
                        img_path, label = t.strip().split(' ', 1)
                        self.img_path.append(dir + img_path + '.jpg')
                        self.label.append([int(x) - 1 for x in label.split(' ')])
                    else:
                        img_path = t.strip()
                        self.img_path.append(dir + img_path + '.jpg')
                        self.label.append([0])
        print('init dataloader done: {}, len = {}'.format(split, len(self.label)))

    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        img_path = self.img_path[index]
        img_label = self.label[index]
#        print(img_path)
        img = Image.open(img_path).convert('RGB')
        label_series = np.zeros(228, dtype=np.float64)
        for i in img_label:
            label_series[i] = 1

        img_label= torch.FloatTensor(label_series)
#        print(img_label)
        if self.transform != None: img = self.transform(img)
#        print(img.size())
#        print(img_label.size())
        return img, img_label
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.img_path)

    def get_number_classes(self):
        return 228


def test_Wish():
    train_data = Wish('hello', 'train')
    test_data = Wish('hello', 'test')
    print(len(train_data))
    print(len(test_data))

if __name__ == "__main__":
    test_Wish()