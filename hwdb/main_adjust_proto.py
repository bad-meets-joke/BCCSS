import torch
import torch.optim as optim
import numpy as np
import os
import logging
import json
from datetime import datetime

from config import opt
from model_adjust_proto_dsbn import AugCMPL
from dataloaders_synhw import *
from train import validate
from utils import *


# ======================================== Setting phase =============================================== 
opt.outf = os.path.join('./results', opt.outf)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# Logger
logger = logging.getLogger('XAO')
logger.setLevel(logging.INFO)
# info_path = opt.outf + '/info_log.txt'
info_path = opt.outf + '/%s_log.txt' % (datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
if not os.path.exists(info_path):
    os.mknod(info_path)
fh = logging.FileHandler(info_path)                  # 输出定向到文件
fh.setLevel(logging.INFO)
sh = logging.StreamHandler()                         # 输出定向到屏幕
sh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

logger.info(json.dumps(vars(opt), indent=4, separators=(',', ': ')))

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

# ======================================== Model =============================================== 
# model
model = AugCMPL(opt)
model = model.cuda()

# save model
modelFilename = os.path.join(opt.outf, opt.saveFilename)

# load checkpoint
checkpoint_path = os.path.join(opt.outf, opt.checkpoint)
if os.path.isfile(checkpoint_path):
    logger.info("Model Loading %s " % checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path)) 

# optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))  # 训练时lr没动
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# dataset
if not opt.test:
    logger.info("Prepare training dataset...")
    trainset = HWDB(opt, is_valid=False)

logger.info("Prepare test dataset...")
testset = HWDB(opt, is_valid=True)

# ======================================== Test =============================================== 
if opt.test:
    logger.info('\n............Start test...........\n')
    
    # Real seen/unseen from HWDB1.0-1.1.
    # unseen仅作评估
    trnset = HWDB(opt, is_valid=False)
    logger.info('Load training set')
    
    # Syn
    synset_unseen = KShotSynHWDB(opt.syn_path, opt, is_valid=True)                                      
    logger.info('Load unseen synthesized set: %s' % opt.syn_path)
    logger.info('The synthesized sample size: %d-shot' % opt.syn_kshot)
    
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=opt.testBatchSize, shuffle=False, 
        num_workers=int(opt.workers), drop_last=False, pin_memory=True)

    model.eval()

    if not opt.bayes:
        """原型分类器校准"""
        INTERPOLATION = opt.interpolation1  # 靠近合成均值的程度
        logger.info('The interpolation value: %.2f' % INTERPOLATION)
        logger.info('Calibarating the prototype')
        model.finetune_prototype(synset_unseen.kshot_unseen, interpolation=INTERPOLATION, k=opt.syn_kshot)

        if not opt.gzsl:
            # ZSL
            epoch = 0
            prec_t, = validate(test_loader, model, epoch, logger, opt)
            logger.info('The accuracy of test: {acc_test:.5f}\t in ZSL'.format(acc_test=prec_t))
            logger.info('-----------------------------------')
        else:
            # GZSL
            opt.gzsl = True
            prec_t, = validate(test_loader, model, epoch, logger, opt)
            logger.info('The accuracy of test: {acc_test:.5f}\t in GZSL'.format(acc_test=prec_t))
            logger.info('-----------------------------------')
    else:
        """贝叶斯分类器校准"""
        interpolation1 = opt.interpolation1  # 用于RDA         
        alpha = opt.alpha                    # 同上
        beta = opt.beta                      # 同上
        knn = 10                       # 用于local smoothing
        alpha2 = 1.0                   # 同上

        if opt.cal_mu == 'no':
            logger.info('[%s, %s]' % (opt.cal_mu, opt.cal_cov)) 
        elif opt.cal_cov in ['RDA', 'RDA-S', 'RDA-A']:
            logger.info('[%s, %s]. interpolation 1: %.2f. alpha: %.2f. beta: %.2f' % (opt.cal_mu, opt.cal_cov, interpolation1, alpha, beta))  # U
        elif opt.cal_cov in ['LSQDF', 'LSQDF-S', 'LSQDF-A']:
            logger.info('[%s, %s]. interpolation 1: %.2f. KNN: %d. alpha2: %.2f. ' % (opt.cal_mu, opt.cal_cov, interpolation1, knn, alpha2))
        elif opt.cal_cov in ['QDF+LDF', 'QDF+LDF-S', 'QDF+LDF-A']:
            logger.info('[%s, %s]. interpolation 1: %.2f. alpha: %.2f.' % (opt.cal_mu, opt.cal_cov, interpolation1, alpha))                   # U                
        elif opt.cal_cov in ['QDF', 'LDF', 'LDF-S', 'LDF-A', 'no', 'ideal']:
            logger.info('[%s, %s]. interpolation 1: %.2f.' % (opt.cal_mu, opt.cal_cov, interpolation1))
        else:
            raise ValueError()
        
        logger.info('Estimating the params of bayes classifier with %d SHOT' % opt.syn_kshot)
        
        model.bayes_param_estimate(
            synset_unseen.kshot_unseen, trnset.seen, trnset.unseen, k=opt.syn_kshot, 
            interpolation1=interpolation1, 
            alpha=alpha, beta=beta, 
            knn=knn, alpha2=alpha2
        )
        logger.info("done!")
        
        if not opt.gzsl:
            # ZSL
            epoch = 0
            prec_t, = validate(test_loader, model, epoch, logger, opt)
            logger.info('The accuracy of test: {acc_test:.5f}\t in ZSL'.format(acc_test=prec_t))
            logger.info('-----------------------------------')
        else:
            # GZSL
            epoch = 0
            prec_t, = validate(test_loader, model, epoch, logger, opt)
            logger.info('The accuracy of test: {acc_test:.5f}\t in ZSL'.format(acc_test=prec_t))
            logger.info('-----------------------------------')


