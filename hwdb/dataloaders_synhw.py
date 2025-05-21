import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

from config import opt


class KShotSynHWDB(Dataset):
    """合成的未见类数据集。每个未见类有一些样本, 用于微调原型"""
    def __init__(self, db_path, opt, is_valid=True):
        is_valid = True  # 注意，仅unseen synthesized data
        dname = 'syn'
        with h5py.File(db_path, 'r') as f: 
            self.x = f[dname + '/x'][:]      # |U|*K*1*H*W
            self.idc = f[dname + '/idc'][:]  # |U|*K*1
        
        # build trn or tst subset from 3755
        np.random.seed(opt.randomseed)
        idx = np.random.permutation(3755)
        seenSize = opt.seenSize
        if is_valid:
            index = idx[2755:]      # characters for test
        else:
            index = idx[:seenSize]  # characters for training
        is_choose = np.where(np.sum(self.idc.reshape((-1, 1)) == index, axis=1))[0]
        assert len(is_choose) == self.idc.reshape(-1).shape[0]

        kshot_samples = self.x
        kshot_samples = 1. - kshot_samples/255.
        kshot_samples = torch.from_numpy(kshot_samples.astype('float32'))
        self.kshot_unseen = kshot_samples

        self.len = self.x.shape[0]     

    def __getitem__(self, index):
        x = 1. - self.x[index] / 255.0  # background of 0, foreground of 1.
        idc = self.idc[index]      
        return x.astype('float32'), idc.astype('long')

    def __len__(self):
        return self.len


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
            # Load dataset
            if opt.input_size == 64:
                db_path = 'HWDB_subset_3755.hdf5'
            else:
                db_path = 'HWDB_subset_3755_32x32.hdf5'
            dname = 'trn' if not is_valid else 'tst'  # 原始的训练集或测试集
            with h5py.File(db_path, 'r') as f: 
                self.x = f[dname + '/x'][:]
                self.tag = f[dname + '/tagcode'][:]
                self.idc = f[dname + '/idc'][:]

            # build trn or tst subset from 3755
            np.random.seed(opt.randomseed)
            idx = np.random.permutation(3755)  # 常用3755类在前面！
            seenSize = opt.seenSize
            if is_valid:
                index = idx[2755:]      # characters for test
            else:
                index = idx[:seenSize]  # characters for training
            is_choose = np.where(np.sum(self.idc == index, axis=1))[0]
            
            if not is_valid:
                # real seen from HWDB1.0-1.1
                seen = []
                seen_index = idx[:seenSize]
                trn_idc = self.idc.reshape(-1)
                for i in seen_index:
                    samples = self.x[trn_idc==i]
                    samples = 1. - samples/255.
                    samples = torch.from_numpy(samples.astype('float32'))
                    seen.append(samples)
                self.seen = seen
                
                # real unseen form HWDB1.0-1.1
                unseen = []
                unseen_index = idx[2755:]
                trn_idc = self.idc.reshape(-1)
                for i in unseen_index:
                    samples = self.x[trn_idc==i]
                    samples = 1. - samples/255.
                    samples = torch.from_numpy(samples.astype('float32'))
                    unseen.append(samples)
                self.unseen = unseen

            self.x = self.x[is_choose]
            self.tag = self.tag[is_choose]
            self.idc = self.idc[is_choose]
        elif opt.task == 2: # HWDB 1.0,1,1(3755) >> ICDAR2013(3755)
            dname = 'trn' if not is_valid else 'tst'
            with h5py.File('HWDB_subset_3755.hdf5', 'r') as f: 
                self.x = f[dname + '/x'][:]
                self.tag = f[dname + '/tagcode'][:]
                self.idc = f[dname + '/idc'][:]
        else: # Same as opt.task==1
            dname = 'trn' if not is_valid else 'tst'  # 原始的训练集或测试集
            with h5py.File('HWDB_subset_3755.hdf5', 'r') as f: 
                self.x = f[dname + '/x'][:]
                self.tag = f[dname + '/tagcode'][:]
                self.idc = f[dname + '/idc'][:]

            # build trn or tst subset from 3755
            np.random.seed(opt.randomseed)
            idx = np.random.permutation(3755)  # 常用3755类在前面！
            seenSize = opt.seenSize
            if is_valid:
                index = idx[2755:]      # characters for test
            else:
                index = idx[:seenSize]  # characters for training
            is_choose = np.where(np.sum(self.idc == index, axis=1))[0]

            self.x = self.x[is_choose]
            self.tag = self.tag[is_choose]
            self.idc = self.idc[is_choose]
        
        self.len = self.tag.shape[0]     

    def __getitem__(self, index):
        x = 1. - self.x[index] / 255.0  # background of 0, foreground of 1.
        
        tag = self.tag[index]
        idc = self.idc[index]      
        return x.astype('float32'), idc.astype('long')

    def __len__(self):
        return self.len

