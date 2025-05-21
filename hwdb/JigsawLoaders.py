import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import h5py
import utils
import pdb
import numpy as np
import matplotlib.pyplot as plt
from config import opt
from random import random
import os


class HWDB(Dataset):
    def __init__(self, opt, is_valid=False):
        if opt.task == 3: # HWDB 1.0,1,1(3755) >> HWDB 1.2(3008)
            if not is_valid:
                with h5py.File('HWDB_subset_3755.hdf5', 'r') as f: # 
                    self.x = np.concatenate([f['trn/x'][:], f['tst/x'][:]], axis=0)
                    self.tag = np.concatenate([f['trn/tagcode'][:], f['tst/tagcode'][:]], axis=0) 
                    self.idc = np.concatenate([f['trn/idc'][:], f['tst/idc'][:]], axis=0)
            elif is_valid:
                with h5py.File('HWDB_subset_3008.hdf5', 'r') as f: # 
                    self.x = f['tst/x'][:]
                    self.tag = f['tst/tagcode'][:]
                    self.idc = f['tst/idc'][:]
        if opt.task == 4: # HWDB 1.0,1,1(3755) >> HWDB 1.2(3008)
            with h5py.File('HWDB_subset_3755.hdf5', 'r') as f1, \
                 h5py.File('HWDB_subset_3008.hdf5', 'r') as f2: #
                self.x = np.concatenate([f1['trn/x'][:], f1['tst/x'][:], f2['tst/x'][:]], axis=0)
                self.tag = np.concatenate([f1['trn/tagcode'][:], f1['tst/tagcode'][:], f2['tst/tagcode'][:]], axis=0) 
                self.idc = np.concatenate([f1['trn/idc'][:], f1['tst/idc'][:], f2['tst/idc'][:]], axis=0)
            
            np.random.seed(opt.randomseed)
            idx = np.random.permutation(6763)   
            n = int(6763 * opt.rate)
            if is_valid:
                index = idx[n:]
            else:
                index = idx[:n] 
            is_choose = np.where(np.sum(self.idc == index, axis=1))[0]
            self.x = self.x[is_choose]
            self.tag = self.tag[is_choose]
            self.idc = self.idc[is_choose]
        elif opt.task == 0: # HWDB 1.0,1.1(2755/all) >> ICDAR2013(1000/all)
            dname = 'trn' if not is_valid else 'tst'
            with h5py.File('HWDB_subset_3755.hdf5', 'r') as f: 
                self.x = f[dname + '/x'][:]
                self.tag = f[dname + '/tagcode'][:]
                self.idc = f[dname + '/idc'][:]
                
            if is_valid:
                index = np.where(self.idc >= 2755)[0]
            else:
                index = np.where(self.idc < 2755)[0]

            self.x = self.x[index]
            self.tag = self.tag[index]
            self.idc = self.idc[index]
        elif opt.task == 1: # HWDB 1.0,1.1(random2755/all) >> ICDAR2013(random1000/all)
            dname = 'trn' if not is_valid else 'tst'
            with h5py.File('HWDB_subset_3755.hdf5', 'r') as f: 
                self.x = f[dname + '/x'][:]
                self.tag = f[dname + '/tagcode'][:]
                self.idc = f[dname + '/idc'][:]
            # build trn or tst subset from 3755
            np.random.seed(opt.randomseed)
            idx = np.random.permutation(3755)  # 常用3755类在前面！
            # if is_valid:
            #     index = idx[2755:]
            # else:
            #     index = idx[:2755]
            seenSize = opt.seenSize
            if is_valid:
                index = idx[2755:]      # characters for test
            else:
                index = idx[:seenSize]  # characters for training
            is_choose = np.where(np.sum(self.idc == index, axis=1))[0]
            
            self.x = self.x[is_choose]
            self.tag = self.tag[is_choose]
            self.idc = self.idc[is_choose]
        elif opt.task == 2: # HWDB 1.0,1,1(3755) >> ICDAR2013(3755)
            dname = 'trn' if not is_valid else 'tst'
            with h5py.File('HWDB_subset_3755.hdf5', 'r') as f: 
                self.x = f[dname + '/x'][:]
                self.tag = f[dname + '/tagcode'][:]
                self.idc = f[dname + '/idc'][:]

        self.len = self.tag.shape[0]     
        # ----------------------------------------------
        self.is_valid = is_valid
        self.grid_size = 3
        self.jig_classes = opt.jigsaw_n_classes
        self.permutations = self.__retrieve_permutations(self.jig_classes)  # 30x9
        self.bias_whole_img = opt.bias_whole_img
        
        def make_grid(x):
            # return torchvision.utils.make_grid(x, self.grid_size, padding=0)
            v1 = np.concatenate(x[0:3], 1)
            v2 = np.concatenate(x[3:6], 1)    
            v3 = np.concatenate(x[6:9], 1)    
            x = np.concatenate((v1, v2, v3), 0)
            return x
        self.returnFunc = make_grid
        
    def get_tile(self, img, n):
        w = int(img.shape[1] / self.grid_size)
        x = int(n / self.grid_size)
        y = n % self.grid_size
        tile = img[x*w:(x+1)*w, y*w:(y+1)*w]
        # tile = self._augment_tile(tile)
        return tile

    def __retrieve_permutations(self, classes):
        all_perm = np.load(os.path.join('hwdb/jigsaw_permutations', 'permutations_%d.npy' % (classes)))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm

    def __getitem__(self, index):
        # x = self.x[index] / 255.0     # <<<-------normalization for image!
        x = 1. - self.x[index] / 255.0  # 背景0, 前景1, 1xHxW
        # tag = self.tag[index]
        idc = self.idc[index]      
        
        if not self.is_valid:
            # 仅训练jigsaw
            # 裁剪。保证可整除grid
            h, w = x.shape[1], x.shape[2]
            new_h, new_w = int(h / self.grid_size) * self.grid_size, int(w / self.grid_size) * self.grid_size  # 可整除
            res_h, res_w = h - new_h, w - new_w 
            img = x[0][:new_h, :new_w]
            
            # 分块
            n_grids = self.grid_size ** 2
            tiles = [None] * n_grids
            for n in range(n_grids):
                tiles[n] = self.get_tile(img, n)
            
            # 随机shuffle
            order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
            if self.bias_whole_img:
                if self.bias_whole_img > random():
                    order = 0
            if order == 0:
                data = tiles
            else:
                data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]    
            
            # 重组
            data = self.returnFunc(data)
            
            # padding到正常尺寸
            data = np.lib.pad(data, ((0, res_h), (0, res_w)), mode='constant', constant_values=0.)            
            
            # 加上通道
            x = np.expand_dims(data, 0)

            order = int(order)
        else:
            # 测试不用jiasaw
            x = x
            order = int(0)
        return x.astype('float32'), order, idc.astype('long')

    def __len__(self):
        """Necessary method!!!"""
        return self.len


if __name__ == '__main__':
    allowed_set = utils.get_coding_character('gb2312')

    for is_valid in (True, False):
        dataset = HWDB(opt, is_valid)
        pdb.set_trace()

        data_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=256, shuffle=True, 
                        num_workers=8, pin_memory=True)
        pdb.set_trace()

        image = np.zeros((16*64, 16*64), dtype=np.uint8)
        for episode, (x, y) in enumerate(data_loader):
            # pdb.set_trace()
            B, _ = y.shape 
            pdb.set_trace()
            cht_list = []
            if (B == 256):
                for i in range(16):
                    for j in range(16):
                        idx = 16*i + j
                        image[i*64 : (i+1) * 64, j*64 : (j+1) * 64] = 255 * x[idx][0]
                        cht_list.append(allowed_set[y[idx]])

            print (cht_list)
            plt.imsave('test_loader.png', image)





