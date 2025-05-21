"""把backbone(基于HDCE)改成DSBN形式, 为提高整体精度
"""

import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import math
import torch.nn.functional as F
import utils
import os
import numpy as np
import h5py
import cv2
from torchvision import transforms

from resnetlike import ResNetLike, ResNetLikeV2, ResNetLikeV3, ResNetLikeV4



class ImgtoClass_Metric(nn.Module):
    def __init__(self, opt):
        super(ImgtoClass_Metric, self).__init__()
        self.distance = opt.distance_prn
         
    def cal_cosinesimilarity(self, input1, input2):
        # pdb.set_trace()
        B, C = input1.size()
        B2, C2 = input2.size()
        assert(C == C2)
        query_sam = input1
        query_sam_norm = torch.norm(query_sam, p=2, dim=1, keepdim=True)

        support_sam = torch.transpose(input2, 0, 1) # C * B2
        support_sam_norm = torch.norm(support_sam, p=2, dim=0, keepdim=True)

        Similarity_matrix = query_sam@support_sam / (query_sam_norm@support_sam_norm) # B * B2
        return Similarity_matrix

    def cal_pdist(self, input1, input2, p=2):
        B, C = input1.size()
        B2, C2 = input2.size()
        assert(C == C2)
        pdist = -1 * torch.norm(input1.unsqueeze(1) - input2.unsqueeze(0), p=p, dim=2)   # 欧氏距离
        return pdist    

    def cal_pdist_square(self, input1, input2, p=2):
        B, C = input1.size()
        B2, C2 = input2.size()
        assert(C == C2)
        pdist = -1 * (input1.unsqueeze(1) - input2.unsqueeze(0)).pow(2).sum(2)         # 欧氏距离的平方, 标准DCE
        return pdist    

    def forward(self, x1, x2):
        if self.distance == 'cosine':
            Similarity = self.cal_cosinesimilarity(x1, x2)
        elif self.distance == 'euclidean':
            Similarity = self.cal_pdist(x1, x2)
        elif self.distance == 'euclidean_square':
            Similarity = self.cal_pdist_square(x1, x2)
        else:
            Similarity = None
        return Similarity


class AugCMPL(nn.Module):
    def __init__(self, opt):
        super(AugCMPL, self).__init__()
        self.opt = opt
        
        self.chinese_6763_chars = utils.get_coding_character()

        # backbone
        if opt.backbone == 'resnetlike':
            self.feat_extractor = ResNetLike(opt).cuda()   # cuda()冗余 
        elif opt.backbone == 'resnetlike-v2':
            self.feat_extractor = ResNetLikeV2(opt).cuda()   
        elif opt.backbone == 'resnetlike-v3':
            self.feat_extractor = ResNetLikeV3(opt).cuda()   
        elif opt.backbone == 'resnetlike-v4':
            self.feat_extractor = ResNetLikeV4(opt).cuda()   
        else:
            raise ValueError()

        # similarity metric
        self.imgtoclass_prn = ImgtoClass_Metric(opt).cuda()
        
        # prn imgs
        self.printed_charimgs = self.get_printed_charimgs()  # 6763*64*64. The order is consistent with 'y_list'  

        if opt.scale_choice == 'learnable':
            self.scale_weight2 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.scale_weight2.data.fill_(1.0)
        elif opt.scale_choice == 'constant': 
            self.scale_weight2 = opt.scale_weight
        else:
            raise ValueError()

        if opt.task == 0: # 0:2755-1000 1:2755-1000(random) 2: 3755-3755, 3: 3755-3008
            self.y_trn = torch.arange(1, 2755, dtype=torch.long).cuda()
            self.y_tst = torch.arange(2755, 3755, dtype=torch.long).cuda()
        elif opt.task == 1:
            np.random.seed(opt.randomseed)
            idx = np.random.permutation(3755)
            seenSize = opt.seenSize   
            self.y_trn = torch.tensor(idx[:seenSize], dtype=torch.long).cuda()
            self.y_tst = torch.tensor(idx[2755:], dtype=torch.long).cuda()
            # self.y_tst = torch.tensor(idx[3745:], dtype=torch.long).cuda()
        elif opt.task == 2:
            self.y_trn = torch.arange(1, 3755, dtype=torch.long).cuda()
            self.y_tst = torch.arange(1, 3755, dtype=torch.long).cuda()
        elif opt.task == 3:
            self.y_trn = torch.arange(1, 3755, dtype=torch.long).cuda()
            self.y_tst = torch.arange(3755, 6763, dtype=torch.long).cuda()
        elif opt.task == 4:
            np.random.seed(opt.randomseed)
            idx = np.random.permutation(6763)   
            n = int(6763 * opt.rate)
            self.y_trn = torch.tensor(idx[:n], dtype=torch.long).cuda()
            self.y_tst = torch.tensor(idx[n:], dtype=torch.long).cuda()            
        # pdb.set_trace()

        alpha = [1.0/(i+1) for i in range(1+6763)]
        self.L = torch.tensor([sum(alpha[:i]) for i in range(1+6763)]).cuda()

    def get_printed_charimgs(self):
        # load data and normalize
        if self.opt.input_size == 64:
            prn_db_path = 'printed.hdf5'
        else:
            prn_db_path = 'printed_32x32.hdf5'
        with h5py.File(prn_db_path, 'r') as f: 
            printed_charimgs = 1 - f['printed/x'][:] / 255.0  # background of 0, foreground of 1.
        # tensor
        prn_ci = torch.from_numpy(printed_charimgs.astype('float32')).cuda()
        return prn_ci

    def myloss(self, y_pred, y):
        if self.opt.loss == 3:
            loss_function = nn.CrossEntropyLoss().cuda()
            loss = loss_function(y_pred, y)
        elif self.opt.loss == 2:
            sim_positive = y_pred[:, y.long()].diag()
            sim_top2, idc_top2 = torch.topk(y_pred, k=2, dim=1)
            sim_negative = sim_top2[:, 0]
            right_index = (idc_top2[:, 0] == y)
            sim_negative[right_index] = sim_top2[right_index, 1]
            loss = torch.mean(F.relu(sim_negative - sim_positive + 1))
        elif self.opt.loss == 1:
            sim_positive = y_pred[:, y.long()].diag()
            sim_negative, _ = torch.max(y_pred,dim=1)
            loss = torch.mean(1.0 / (1 + torch.exp(sim_positive - sim_negative)))      
        else :
            sim_positive = y_pred[:, y.long()].diag()
            sim_negative = y_pred
            loss_xy = 1 + sim_negative - sim_positive.unsqueeze(1) # bs * self.negativeSize 3755
            rank = torch.sum(loss_xy>0, dim=1)         # bs
            L_rank = self.L[rank].detach().view(-1, 1) # bs
            loss = torch.mean(L_rank / rank.float() * torch.sum(F.relu(loss_xy), dim=1))
        return loss        

    def predict_partly(self, x, y):
        """ZSL测试"""
        B = x.shape[0]
        # CLOSED or OPEN?
        if (y[0][0] in self.y_tst):
            is_valid = True
        else:
            is_valid = False

        feat = self.feat_extractor(x, 'hw')  # B * C * h * w
        input1 = feat 
        
        if not is_valid:
            printed_feat = self.feat_extractor(self.printed_charimgs[self.y_trn], 'prn')  
            input2 = printed_feat
        else:
            printed_feat = self.feat_extractor(self.printed_charimgs[self.y_tst], 'prn')
            input2 = printed_feat

        M = input2.shape[0]
        sim2 = self.imgtoclass_prn(input1.reshape(B, -1), input2.reshape(M, -1))     #  metric
        y_pred2 = self.scale_weight2 * sim2

        _, idx = torch.max(y_pred2, dim=1)
        if not is_valid:
            idx = self.y_trn[idx]
        else:
            idx = self.y_tst[idx]
        acc = torch.sum(idx == y.view(-1)).float() / B
        correct_count = torch.sum(idx == y.view(-1)).float()        
        wrong_id = torch.where(idx != y.view(-1))[0]  # 错分样本索引，可能为空
        
        return idx, acc, correct_count, wrong_id

    def predict_generalized(self, x, y):
        """广义测试"""
        B = x.shape[0]

        feat = self.feat_extractor(x)  # B * C * h * w
        input1 = feat 
        
        self.seen_proto = self.feat_extractor(self.printed_charimgs[self.y_trn])
        self.unseen_proto = self.feat_extractor(self.printed_charimgs[self.y_tst])
        input2 = torch.cat([self.seen_proto, self.unseen_proto], 0)  # ***用手写更新后的原型***
        self.y_proto = torch.cat([self.y_trn, self.y_tst], 0)        

        M = input2.shape[0]
        sim2 = self.imgtoclass_prn(input1.reshape(B, -1), input2.reshape(M, -1))     #  metric
        y_pred2 = self.scale_weight2 * sim2

        _, idx = torch.max(y_pred2, dim=1)
        idx = self.y_proto[idx]  
        acc = torch.sum(idx == y.view(-1)).float() / B
        correct_count = torch.sum(idx == y.view(-1)).float()        
        wrong_id = torch.where(idx != y.view(-1))[0]  # 错分样本索引，可能为空
        
        return idx, acc, correct_count, wrong_id

    def forward(self, x, y):
        # hw
        B, _ = y.shape
        new_y = torch.nonzero(y==self.y_trn)[:, 1]  # new label id in seen categories
        feat = self.feat_extractor(x, 'hw')         # B * C * h * w (256 * 512 * 4 * 4)
        input1 = feat                    

        # prn
        if self.opt.seenSize <= 500 or self.opt.input_size == 32:
            printed_feat = self.feat_extractor(self.printed_charimgs[self.y_trn], 'prn')
            prn_smp_y = self.y_trn
            prn_smp_y_np = self.y_trn.cpu().detach().numpy()
            new_y_prn = new_y
        else:
            # Sample prn to save GPU memory
            # smp_size = int(self.opt.samplingRate * self.opt.seenSize)
            smp_size = self.opt.samplingSize
            y_trn = self.y_trn.cpu().detach().numpy()
            uniq_y = np.unique(y.cpu().numpy())    # hold true classes
            res = np.setdiff1d(y_trn, uniq_y)
            res_smp_size = smp_size - uniq_y.shape[0]   
            res_smp_y = np.random.choice(res, res_smp_size, replace=False)
            prn_smp_y_np = np.sort(np.concatenate([uniq_y, res_smp_y], 0)) 
            prn_smp_y = torch.from_numpy(prn_smp_y_np).cuda() 
            
            printed_feat = self.feat_extractor(self.printed_charimgs[prn_smp_y], 'prn')
            new_y_prn = torch.nonzero(y==prn_smp_y)[:, 1]
        M = prn_smp_y.shape[0]  # number of sampled printed templates
        input2 = printed_feat         
        
        # dce loss (embedding space, hw --- prn)
        sim2 = self.imgtoclass_prn(input1.reshape(B, -1), input2.reshape(M, -1))  
        y_pred2 = self.scale_weight2 * sim2            
        loss_dce = self.myloss(y_pred2, new_y_prn)     # CE loss
        
        # loss
        loss = loss_dce
        
        # cls acc in a batch
        _, idx = torch.max(y_pred2, dim=1)
        acc = torch.sum(idx == new_y_prn).float() / B  # depend on PRINTED

        output = {
            'loss': loss,
            'loss_dce': loss_dce,
            'batch_acc': acc,
        }

        return output
