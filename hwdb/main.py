import torch
import torch.optim as optim
import numpy as np
from config import opt
import os
import logging
import json
from datetime import datetime

from model_cmpl_dsbn import AugCMPL
from dataloaders import HWDB
from train import training, validate
from utils import *


opt.outf = os.path.join('./results', opt.outf)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# Logger
logger = logging.getLogger('XAO')
logger.setLevel(logging.INFO)
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

# model
model = AugCMPL(opt)
model = model.cuda()

# Save checkpoint
modelFilename = os.path.join(opt.outf, opt.saveFilename)

# Load checkpoint
checkpoint_path = os.path.join(opt.outf, opt.checkpoint)
if opt.test and os.path.isfile(checkpoint_path):
    logger.info("Model Loading from %s " % checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path)) 
else:
    logger.info('Train Model from Scratch')

# optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))  # 训练时lr没动
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# dataset
if not opt.test:
    logger.info("Loading training dataset...")
    trainset = HWDB(opt, is_valid=False)

logger.info("Loading test dataset...")
testset_seen = HWDB(opt, is_valid=True, seen_or_unseen='seen')
testset_unseen = HWDB(opt, is_valid=True, seen_or_unseen='unseen')

logger.info("Trainset size / Testset Seen size / Testset Unseen size: %d / %d / %d" % (len(trainset), len(testset_seen), len(testset_unseen)))


#  Dataloader 
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=opt.trainBatchSize, shuffle=True, 
    num_workers=int(opt.workers), drop_last=True, pin_memory=False
)
test_loader1 = torch.utils.data.DataLoader(
    testset_seen, batch_size=opt.testBatchSize, shuffle=False, 
    num_workers=int(opt.workers), drop_last=False, pin_memory=True
)  # seen 
test_loader2 = torch.utils.data.DataLoader(
    testset_unseen, batch_size=opt.testBatchSize, shuffle=False, 
    num_workers=int(opt.workers), drop_last=False, pin_memory=True
)  # unseen

logger.info('\n............Start training............\n')
start_time = time.time()

best_prec = 0
for epoch in range(opt.epochs):
    logger.info('Trainset: %d' % len(trainset))
    logger.info('Testset seen: %d' % len(testset_seen))
    logger.info('Testset unseen: %d' % len(testset_unseen))

    # train
    training(train_loader, model, optimizer, epoch, logger)  # training one epoch    
    torch.save(model.state_dict(), modelFilename + str(epoch))

    # test
    logger.info('============ Testing on the test set ============')
    prec_t1, = validate(test_loader1, model, epoch, logger, opt)  # seen in seen
    prec_t2, = validate(test_loader2, model, epoch, logger, opt)  # unseen in unseen
    logger.info('the test accuracy in seen/unseen: %.4f / %.4f' % (prec_t1, prec_t2))    
    logger.info('==================================================')

    # lr scheduler
    scheduler.step()
    


