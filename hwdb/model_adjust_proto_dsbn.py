import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import math
import torch.nn.functional as F
import os
import numpy as np
import scipy
import h5py
import cv2
from tqdm import tqdm

import utils
from resnetlike import ResNetLike, ResNetLikeV2, ResNetLikeV3, ResNetLikeV4


class ImgtoClass_Metric(nn.Module):
    def __init__(self, distance):
        """Similarity in same space"""
        super(ImgtoClass_Metric, self).__init__()
        self.distance = distance
         
    def cal_cosinesimilarity(self, input1, input2):
        # pdb.set_trace()
        B, C = input1.size()
        B2, C2 = input2.size()
        assert(C == C2)
        query_sam = input1
        query_sam_norm = torch.norm(query_sam, p=2, dim=1, keepdim=True)

        support_sam = torch.transpose(input2, 0, 1) # C * B2
        support_sam_norm = torch.norm(support_sam, p=2, dim=0, keepdim=True)

        Similarity_matrix = query_sam@support_sam / (query_sam_norm@support_sam_norm)  # B * B2
        return Similarity_matrix

    def cal_pdist(self, input1, input2, p=2):
        B, C = input1.size()
        B2, C2 = input2.size()
        assert(C == C2)
        pdist = -1 * torch.norm(input1.unsqueeze(1) - input2.unsqueeze(0), p=p, dim=2)  # 欧式距离
        return pdist

    def cal_innerdot(self, input1, input2):
        # pdb.set_trace()
        B, C = input1.size()
        B2, C2 = input2.size()
        assert(C == C2)
        query_sam = input1
        support_sam = torch.transpose(input2, 0, 1) # C * B2
        Similarity_matrix = query_sam@support_sam  # B * B2
        return Similarity_matrix   

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
        self.imgtoclass_prn = ImgtoClass_Metric(opt.distance_prn).cuda()
        
        # prn imgs
        self.printed_charimgs = self.get_printed_charimgs()  # 6763 * 64 * 64. The order is consistent with 'y_list'  
        
        if opt.scale_choice == 'learnable':
            self.scale_weight2 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.scale_weight2.data.fill_(1.0)
        elif opt.scale_choice == 'constant': 
            self.scale_weight2 = opt.scale_weight
        else:
            raise ValueError()
        
        # 类别获取。用随机种子来控制和dataset的划分的类别一致
        if opt.task == 0: # 0:2755-1000 1:2755-1000(random) 2: 3755-3755, 3: 3755-3008
            self.y_trn = torch.arange(1, 2755, dtype=torch.long).cuda()
            self.y_tst = torch.arange(2755, 3755, dtype=torch.long).cuda()
        elif opt.task == 1:
            np.random.seed(opt.randomseed)
            idx = np.random.permutation(3755)  
            seenSize = opt.seenSize
            self.idx = idx   
            self.y_trn = torch.tensor(idx[:seenSize], dtype=torch.long).cuda()
            self.y_tst = torch.tensor(idx[2755:], dtype=torch.long).cuda()  # 最后1000类用作测试
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

        # 原型校正
        self.syn_mean = None

        # 贝叶斯分类器 (Gaussian)
        self.syn_mu = None
        self.syn_kshot = None
        self.real_seen_mu = None
        self.real_unseen_mu = None


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
            # pdb.set_trace()
            sim_negative[right_index] = sim_top2[right_index, 1]
            loss = torch.mean(F.relu(sim_negative - sim_positive + 1))
        elif self.opt.loss == 1:
            sim_positive = y_pred[:, y.long()].diag()
            sim_negative, _ = torch.max(y_pred,dim=1)
            # pdb.set_trace()
            loss = torch.mean(1.0 / (1 + torch.exp(sim_positive - sim_negative)))      
        else :
            sim_positive = y_pred[:, y.long()].diag()
            sim_negative = y_pred
            # pdb.set_trace()
            loss_xy = 1 + sim_negative - sim_positive.unsqueeze(1) # bs * self.negativeSize 3755
            rank = torch.sum(loss_xy > 0, dim=1) # bs
            L_rank = self.L[rank].detach().view(-1, 1) # bs

            loss = torch.mean(L_rank / rank.float() * torch.sum(F.relu(loss_xy), dim=1))
        return loss        


    def finetune_prototype(self, kshot_samples, interpolation=0.5, k=30):
        """
        Args:
            kshot_unseen: a list of unseen samples, k shot for each class
            interpolation: closer to handwritten features with bigger value
        """
        with torch.no_grad():
            printed_feat = self.feat_extractor(self.printed_charimgs[self.y_tst], 'prn')
            prn_proto = printed_feat
            
            if self.syn_mean is None:  # 遍历超参时减少重复计算
                syn_mean = []
                for i in tqdm(range(self.y_tst.shape[0])):
                    mu = self.feat_extractor(kshot_samples[i][:k].cuda(), 'hw').mean(0)
                    syn_mean.append(mu)
                self.syn_mean = torch.stack(syn_mean, 0)
            finetuned_proto = (1-interpolation) * prn_proto + interpolation * self.syn_mean
            
            self.unseen_proto = finetuned_proto
            self.seen_proto = self.feat_extractor(self.printed_charimgs[self.y_trn], 'prn')

    def extend_prototype(self, kshot_samples):
        """
        Args:
            kshot_unseen: a list of unseen samples, k shot each class
        """
        with torch.no_grad():
            printed_feat = self.feat_extractor(self.printed_charimgs[self.y_tst])
            prn_proto = printed_feat  # |U| * d

            extended_proto = []
            for i in range(self.y_tst.shape[0]):
                hw_feat = self.feat_extractor(kshot_samples[i].cuda())                     #  K * d (d也可能为c*h*w)
                extended_proto.append(torch.cat((prn_proto[i].unsqueeze(0), hw_feat), 0))  # (K+1) * d 
            extended_proto = torch.cat(extended_proto, 0)                                  # (|U|*(K+1)) * d

            self.proto = extended_proto
            self.y_proto = self.y_tst.unsqueeze(1).expand(self.y_tst.shape[0], kshot_samples[0].shape[0]+1).reshape(-1)  

    def bayes_param_estimate(self, kshot_unseen, seen, unseen, k=1000, 
                             interpolation1=0.5, 
                             alpha=0, beta=0, 
                             knn=10, alpha2=0.0):
        """对已见类和未见类的特征，进行贝叶斯分类器的参数估计
        
        Args:
            第1行为合成数据, 真实已见类数据, 真实未见类数据（不可用，仅评测），合成样本的个数
            第2行为校正原型或均值的系数
            第3行为RDA校正协方差的系数
            第4行为LSQDF校正协方差的系数

        Return:
            None
        """
        with torch.no_grad():
            # prn prototype
            printed_feat = self.feat_extractor(self.printed_charimgs[self.y_tst], 'prn')
            prn_proto = printed_feat  # |U|*d

            # 高斯分布假设的参数估计
            # unseen, syn
            if self.syn_mu is None or self.syn_kshot != k: 
                syn_mu = []
                syn_delta = []
                for i in tqdm(range(self.y_tst.shape[0])):
                    feat = self.feat_extractor(kshot_unseen[i][:k].cuda(), 'hw')
                    mu = feat.mean(0)
                    delta = (feat - mu).transpose(0, 1) @ (feat-mu) / (feat.size(0))
                    
                    syn_mu.append(feat.mean(0))
                    syn_delta.append(delta)
                self.syn_mu = torch.stack(syn_mu, 0)                # |U|*d
                self.syn_delta = torch.stack(syn_delta, 0)          # |U|*d*d
                self.syn_kshot = k  # flag

            # seen, real, from train
            if self.real_seen_mu is None:
                real_seen_mu = []
                real_seen_delta = []
                for i in tqdm(range(self.y_trn.shape[0])):
                    feat = self.feat_extractor(seen[i].cuda(), 'hw')
                    mu = feat.mean(0)
                    delta = (feat - mu).transpose(0, 1) @ (feat-mu) / (feat.size(0))
                    real_seen_mu.append(feat.mean(0))
                    real_seen_delta.append(delta)
                self.real_seen_mu = torch.stack(real_seen_mu, 0)
                self.real_seen_delta = torch.stack(real_seen_delta, 0)
                
                self.seen_mu = self.real_seen_mu                              
                self.seen_delta = self.real_seen_delta.double()               # 变精度float64, 避免其行列式数值溢出, 而且其逆更准确

                # 协方差的逆
                self.seen_inv_delta = torch.inverse(self.seen_delta).float()  # 协方差矩阵的逆，后面判别函数要用
                assert not torch.isinf(self.seen_inv_delta).any().item()              
                assert not torch.isnan(self.seen_inv_delta).any().item()

            # unseen, real, from train. 仅在评估时使用！
            if self.real_unseen_mu is None:
                real_unseen_mu = []
                real_unseen_delta = []
                for i in tqdm(range(self.y_tst.shape[0])):
                    feat = self.feat_extractor(unseen[i].cuda())
                    mu = feat.mean(0)
                    delta = (feat - mu).transpose(0, 1) @ (feat-mu) / (feat.size(0))
                    
                    real_unseen_mu.append(feat.mean(0))
                    real_unseen_delta.append(delta)
                self.real_unseen_mu = torch.stack(real_unseen_mu, 0)
                self.real_unseen_delta = torch.stack(real_unseen_delta, 0)


            # =====================Calibrate the Mean=========================
            if self.opt.cal_mu == 'no':
                self.unseen_mu = prn_proto
            elif self.opt.cal_mu == 'raw':
                self.unseen_mu = self.syn_mu
            elif self.opt.cal_mu == 'intp':
                self.unseen_mu = (1-interpolation1) * prn_proto + interpolation1 * self.syn_mu  # |U|*d
            elif self.cal_mu == 'ideal':
                self.unseen_mu = self.real_unseen_mu
            else:
                raise ValueError('Wrong way to calibrate the mu')

            # ============================Calibrate the Cov===================================
            U, d = self.unseen_mu.shape[0], self.unseen_mu.shape[1]
            if self.opt.cal_cov == 'no' or self.opt.cal_cov == 'QDF':
                self.unseen_delta = self.syn_delta
            elif self.opt.cal_cov == 'ideal':
                self.unseen_delta = self.real_unseen_delta
            elif self.opt.cal_cov == 'LDF':  
                self.unseen_delta = self.syn_delta.mean(0, keepdim=True).expand(self.unseen_mu.size(0), -1, -1)
            elif self.opt.cal_cov == 'LDF-S':
                self.unseen_delta = self.real_seen_delta.mean(0, keepdim=True).expand(self.unseen_mu.size(0), -1, -1)
            elif self.opt.cal_cov == 'LDF-A':
                all_delta = torch.cat([self.real_seen_delta, self.syn_delta], 0)
                self.unseen_delta = all_delta.mean(0, keepdim=True).expand(self.unseen_mu.size(0), -1, -1)
            
            elif self.opt.cal_cov == 'LSQDF':    # local smoothing in unseen
                self.unseen_delta = torch.empty(U, d, d).cuda()
                sim = self.imgtoclass_prn(self.unseen_mu, self.unseen_mu)     # |U|*|U|
                _, knn_inds = torch.topk(sim, knn+1, 1)                       # |U|*(K+1), 需要排除在第一个的自身
                for i in range(self.unseen_mu.shape[0]):
                    self.unseen_delta[i] = (1 - alpha2) * self.syn_delta[i] + alpha2 * self.syn_delta[knn_inds[i][1:]].mean(0)
            elif self.opt.cal_cov == 'LSQDF-S':  # local smoothing in seen
                self.unseen_delta = torch.empty(U, d, d).cuda()
                sim = self.imgtoclass_prn(self.unseen_mu, self.real_seen_mu)  # |U|*|S|
                _, knn_inds = torch.topk(sim, knn, 1)                         # |U|*K
                for i in range(self.unseen_mu.shape[0]):
                    self.unseen_delta[i] = (1 - alpha2) * self.syn_delta[i] + alpha2 * self.real_seen_delta[knn_inds[i]].mean(0)
            elif self.opt.cal_cov == 'LSQDF-A':  # local smoothing in seen+unseen
                self.unseen_delta = torch.empty(U, d, d).cuda()
                all_mu = torch.cat([self.real_seen_mu, self.unseen_mu], 0)
                all_delta = torch.cat([self.real_seen_delta, self.syn_delta], 0)
                sim = self.imgtoclass_prn(self.unseen_mu, all_mu)             # U*(U+S)
                _, knn_inds = torch.topk(sim, knn+1, 1)                       # U*(K+1), 需排除在第一个的自身
                for i in range(self.unseen_mu.shape[0]):
                    self.unseen_delta[i] = (1 - alpha2) * self.syn_delta[i] + alpha2 * all_delta[knn_inds[i][1:]].mean(0)                

            elif self.opt.cal_cov == 'QDF+LDF':     
                self.unseen_delta = (1 - alpha) * self.syn_delta + alpha * self.syn_delta.mean(0, keepdim=True) 
            elif self.opt.cal_cov == 'QDF+LDF-S':     
                self.unseen_delta = (1 - alpha) * self.syn_delta + alpha * self.real_seen_delta.mean(0, keepdim=True) 
            elif self.opt.cal_cov == 'QDF+LDF-A':
                all_delta = torch.cat([self.real_seen_delta, self.syn_delta], 0)     
                self.unseen_delta = (1 - alpha) * self.syn_delta + alpha * all_delta.mean(0, keepdim=True)
            
            elif self.opt.cal_cov == 'RDA':     
                self.unseen_delta = (1 - alpha) * self.syn_delta + alpha * self.syn_delta.mean(0, keepdim=True) 
                self.unseen_delta = (1 - beta) * self.unseen_delta + \
                                    beta * (1./d) * torch.einsum('bii->b', self.unseen_delta).unsqueeze(1).unsqueeze(2) * \
                                        torch.eye(d).unsqueeze(0).expand(U, -1, -1).cuda()
            elif self.opt.cal_cov == 'RDA-S':    # LDF-S involved                
                self.unseen_delta = (1 - alpha) * self.syn_delta + alpha * self.real_seen_delta.mean(0, keepdim=True)
                self.unseen_delta = (1 - beta) * self.unseen_delta + \
                                    beta * (1./d) * torch.einsum('bii->b', self.unseen_delta).unsqueeze(1).unsqueeze(2) * \
                                        torch.eye(d).unsqueeze(0).expand(U, -1, -1).cuda()
            elif self.opt.cal_cov == 'RDA-A':    # LDF-A involved                
                all_delta = torch.cat((self.real_seen_delta, self.syn_delta), dim=0)
                self.unseen_delta = (1 - alpha) * self.syn_delta + alpha * all_delta.mean(0, keepdim=True)
                self.unseen_delta = (1 - beta) * self.unseen_delta + \
                                    beta * (1./d) * torch.einsum('bii->b', self.unseen_delta).unsqueeze(1).unsqueeze(2) * \
                                        torch.eye(d).unsqueeze(0).expand(U, -1, -1).cuda()
            else:
                raise ValueError()
            
            self.unseen_delta = self.unseen_delta.double()                    # 变精度float64, 避免计算行列式出现溢出, 同时计算逆更准一些
            
            # 协方差的逆
            self.unseen_inv_delta = torch.inverse(self.unseen_delta).float()  # 协方差矩阵的逆，后面判别函数要用
            assert not torch.isinf(self.unseen_inv_delta).any().item()              
            assert not torch.isnan(self.unseen_inv_delta).any().item()

            # 协方差的行列式（用于判别函数）, 除以一个常数C来避免数值溢出。
            # 该数需做精细选择
            if not self.opt.gzsl:
                self.delta_C = select_C(self.unseen_delta)
            else:
                self.delta_C = select_C(torch.cat((self.seen_delta, self.unseen_delta), 0))
     
        return

    def bayes_decision(self, x, y):
        """ZSL"""
        B = x.shape[0]
        # CLOSED or OPEN?
        if (y[0][0] in self.y_tst):
            is_valid = True
        else:
            is_valid = False

        feat = self.feat_extractor(x, 'hw')  
        
        # Discriminant Function  
        ff = (feat.unsqueeze(1) - self.unseen_mu).unsqueeze(2)                         # B*N*1*d
        item1 = -1 *  ff @ self.unseen_inv_delta @ ff.transpose(2, 3)                 # (B*N*1*d)(N*d*d)(B*N*d*1) --> B*N*1*1
        item2 = -1 *  torch.log(torch.det(self.unseen_delta / self.delta_C)).float()  # 除以一个数C是为了避免行列式出现inf。C不能太小（有inf），也不能太大（有0）                   
        assert not (torch.isinf(item2).any().item() or torch.isnan(item2).any().item())
        y_pred2 = item1.squeeze(3).squeeze(2) + item2.unsqueeze(0)

        _, idx = torch.max(y_pred2, dim=1)
        if not is_valid:
            idx = self.y_trn[idx]
        else:
            idx = self.y_tst[idx]  # ***************************

        acc = torch.sum(idx == y.view(-1)).float() / B
        correct_count = torch.sum(idx == y.view(-1)).float()        
        wrong_id = torch.where(idx != y.view(-1))[0]  # 错分样本索引，可能为空      
        
        return idx, acc, correct_count, wrong_id

    def bayes_decision_generalized(self, x, y):
        """GZSL
        
        方式一: 统一贝叶斯，但需要保存训练样本
        方式二: 原型和贝叶斯混用。先用最近原型分类器判断来自S还是U。如果在S, 继续原型分类器, 在U采用贝叶斯分类器。
        
        """
        B = x.shape[0]
        
        # Candidate categories
        self.y_all = torch.cat([self.y_trn, self.y_tst], 0)

        feat = self.feat_extractor(x, 'hw')  
        
        # Discriminant Function 
        # - Unseen part
        ff = (feat.unsqueeze(1) - self.unseen_mu).unsqueeze(2)                      
        item1 = -1 *  ff @ self.unseen_inv_delta @ ff.transpose(2, 3)                
        item2 = -1 *  torch.log(torch.det(self.unseen_delta / self.delta_C)).float()                  
        assert not (torch.isinf(item2).any().item() or torch.isnan(item2).any().item())
        y_pred2 = item1.squeeze(3).squeeze(2) + item2.unsqueeze(0)

        # - Seen part
        ff = (feat.unsqueeze(1) - self.seen_mu).unsqueeze(2)                            
        item1 = -1 * ff @ self.seen_inv_delta @ ff.transpose(2, 3)                       
        item2 = -1 * torch.log(torch.det(self.seen_delta / self.delta_C)).float()                         
        assert not (torch.isinf(item2).any().item() or torch.isnan(item2).any().item())
        y_pred1 = item1.squeeze(3).squeeze(2) + item2.unsqueeze(0)

        y_pred = torch.cat((y_pred1, y_pred2), 1)    # B*(S+U)
        _, idx = torch.max(y_pred, dim=1)
        idx = self.y_all[idx]                        
        
        acc = torch.sum(idx == y.view(-1)).float() / B
        correct_count = torch.sum(idx == y.view(-1)).float()        
        wrong_id = torch.where(idx != y.view(-1))[0]    
        
        return idx, acc, correct_count, wrong_id

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
            input2 = self.unseen_proto 

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
        wrong_id = torch.where(idx != y.view(-1))[0]  
        return idx, acc, correct_count, wrong_id

    def predict_generalized(self, x, y):
        """广义测试"""
        B = x.shape[0]

        feat = self.feat_extractor(x)  # B*C*h*w
        input1 = feat 
        
        self.seen_proto = self.feat_extractor(self.printed_charimgs[self.y_trn])
        input2 = torch.cat([self.seen_proto, self.unseen_proto], 0)  # seen + updated unseen!
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
        """训练阶段"""
        # hw
        B, _ = y.shape
        new_y = torch.nonzero(y==self.y_trn)[:, 1]  # new label id in seen categories
        feat = self.feat_extractor(x, 'hw')         # B * C * h * w (256 * 512 * 4 * 4)
        input1 = feat                    

        # prn
        if self.opt.samplingRate == 1.:
            printed_feat = self.feat_extractor(self.printed_charimgs[self.y_trn], 'prn')
            prn_smp_y = self.y_trn
            prn_smp_y_np = self.y_trn.cpu().detach().numpy()
            new_y_prn = new_y
        else:
            # sample prn to save GPU memory
            smp_size = int(self.opt.samplingRate * self.opt.seenSize)
            y_trn = self.y_trn.cpu().detach().numpy()
            uniq_y = np.unique(y.cpu().numpy())    # hold true classes
            res = np.setdiff1d(y_trn, uniq_y)
            res_smp_size = smp_size - uniq_y.shape[0]   
            res_smp_y = np.random.choice(res, res_smp_size, replace=False)
            prn_smp_y_np = np.sort(np.concatenate(uniq_y, res_smp_y)) 
            prn_smp_y = torch.from_numpy(prn_smp_y_np).cuda() 
            
            printed_feat = self.feat_extractor(self.printed_charimgs[prn_smp_y], 'prn')
            new_y_prn = torch.nonzero(y==prn_smp_y)[:, 1]
        M = prn_smp_y.shape[0]  # number of sampled printed templates
        input2 = printed_feat         
        
        # dce loss (embedding space, hw --- prn)
        sim2 = self.imgtoclass_prn(input1.reshape(B, -1), input2.reshape(M, -1))  
        y_pred2 = self.scale_weight2 * sim2            # scale_weight2
        loss_dce = self.myloss(y_pred2, new_y_prn)  
        
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



def select_C(cov):
    """挑选一个常数, 避免协方差的行列式数值溢出就，毕竟该值用于后面的判别函数"""
    d = cov.shape[-1]
    C0 = math.floor(math.log((1./d) * torch.einsum('bii->b', cov).mean().item(), 10))  # 指数
    if C0 >= 1:
        C1 = int(5 * math.pow(10, C0))
        step = int(math.pow(10, C0-1))
        num = int(C1 / step) + 1
        for C in np.linspace(C1, 0, num):
            has_inf = torch.isinf(torch.det(cov / C)).any().item()
            has_nan = torch.isnan(torch.det(cov / C)).any().item()
            has_zero = torch.any(torch.det(cov / C) <= 0).item()
            if has_inf or has_nan or has_zero: 
                continue
            else:
                print('Finding good C to divide delta. C is %f' % (C))
                break
        if C == 0:
            print('NOT finding good C to divide delta!')
            exit()
    elif C0 < 0:
        C1 = math.pow(10, C0-2)
        C2 = math.pow(10, C0+1) 
        for C in np.linspace(C1, C2, 10**3+1).tolist():
            has_inf = torch.isinf(torch.det(cov / C)).any().item()
            has_nan = torch.isnan(torch.det(cov / C)).any().item()
            has_zero = torch.any(torch.det(cov / C) <= 0).item()
            if has_inf or has_nan or has_zero: 
                continue
            else:
                print('Finding good C to divide delta. C is %f' % (C))
                break              
    else:
        C1 = 10
        for C in range(10, -1, -1):
            has_inf = torch.isinf(torch.det(cov / C)).any().item()
            has_nan = torch.isnan(torch.det(cov / C)).any().item()
            has_zero = torch.any(torch.det(cov / C) <= 0).item()
            if has_inf or has_nan or has_zero: 
                continue
            else:
                print('Finding good C to divide delta. C is %f' % (C))
                break
        if C == 0:
            print('NOT finding good C to divide delta!!!')
            exit()
    return C