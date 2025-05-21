import time
import torch
import torch.nn as nn
import pdb
from PIL import Image
import cv2
from tqdm import tqdm

from config import opt
from utils import *


def training(train_loader, model, optimizer, epoch, logger):
    model.train()
    
    end_time = time.time()

    for episode, (x, y) in enumerate(train_loader): # episode is NOT epoch        
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        
        output = model(x, y)

        loss, loss_dce = output['loss'], output['loss_dce']
        acc = output['batch_acc']

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        episode_time = time.time() - end_time  # time of current batch
        end_time = time.time()

        if (episode % 50 ==1):
            logger.info('Epoch-({0}): [{1}/{2}]\t'
                'Time {episode_time:.3f} \t'
                'Lr {lr:.8f}\t'
                'Loss {loss:.3f}\t'
                'Loss_dce {loss_dce:.3f}\t'
                'Prec@1 {acc:.3f}\t'.format(
                    epoch, episode, len(train_loader), episode_time=100*episode_time,
                    lr=optimizer.state_dict()['param_groups'][0]['lr'],
                    loss=loss.item(), loss_dce=loss_dce.item(),
                    acc=acc.item()))
    return 


def validate(test_loader, model, epoch, logger, opt):
    model.eval()

    end_time = time.time()
    
    correct_count_all = 0
    testset_size = 0
    
    with torch.no_grad():
        for episode, (x, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda()
            if not opt.gzsl:
                # ZSL
                if not opt.bayes:
                    preds, accuracy, correct_count, wrong_id = model.predict_partly(x, y)          # ZSL  (not T1)
                else: 
                    preds, accuracy, correct_count, wrong_id = model.bayes_decision(x, y)          # ZSL w/ Bayes         
            else:
                # GZSL
                if not opt.bayes:
                    preds, accuracy, correct_count, wrong_id = model.predict_generalized(x, y)     # GZSL        
                else:
                    preds, accuracy, correct_count, wrong_id = model.bayes_decision_generalized(x, y)

            episode_time = time.time() - end_time
            end_time = time.time()

            if (episode % 100 ==1):
                logger.info('Valid-({0}): [{1}/{2}]\t'
                    'Time {episode_time:.3f} \t'
                    'Prec@ac {accuracy:.3f}\t'.format(
                        epoch, episode, len(test_loader), episode_time=100*episode_time, 
                        accuracy=accuracy.item()))

            correct_count_all = correct_count_all + correct_count.item()
            testset_size = testset_size + x.size()[0]
        
        logger.info('Testset Size: %d' % testset_size)
        prec = correct_count_all / testset_size
    
    return prec, 

