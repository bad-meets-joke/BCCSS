import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pdb
import numpy as np
import matplotlib.pyplot as plt

from config import opt
import utils


class HWDB(Dataset):
    def __init__(self, opt, is_valid=False, seen_or_unseen=None):
        """
        Args:
            is_valid: True for testset, False for trainset
            seen_or_unseen: 'seen' for seen classes. 'unseen' for unseen classes. 'unseen' only works when is_valid=True
        """
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
        elif opt.task == 1 or opt.task == 10 or opt.task == 11 or opt.task == 12 or opt.task == 13: # HWDB 1.0,1.1(random2755/all) >> ICDAR2013(random1000/all)
            # Load dataset
            if opt.input_size == 64:
                db_path = 'HWDB_subset_3755.hdf5'
            else:
                db_path = 'HWDB_subset_3755_32x32.hdf5'
            dname = 'trn' if not is_valid else 'tst'
            with h5py.File(db_path, 'r') as f: 
                self.x = f[dname + '/x'][:]
                self.tag = f[dname + '/tagcode'][:]
                self.idc = f[dname + '/idc'][:]

            # Randomly split trn and tst from 3755 classes
            np.random.seed(opt.randomseed)     # 种子控制seen/unseen类别划分
            idx = np.random.permutation(3755)  # 常用3755类在前面！
            seenSize = opt.seenSize
            if is_valid and seen_or_unseen == 'unseen':
                index = idx[2755:]      # characters for test
                self.y_unseen = index
            else:
                index = idx[:seenSize]  # characters for training
                self.y_seen = index
            
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

    def __getitem__(self, index):
        x = 1. - self.x[index] / 255.0  # background of 0, foreground of 1.
        tag = self.tag[index]
        idc = self.idc[index]      
        return x.astype('float32'), idc.astype('long')

    def __len__(self):
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





