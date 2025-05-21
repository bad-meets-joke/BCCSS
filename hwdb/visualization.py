import torch
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import rgb2hex
from tqdm import tqdm
import pdb


def visualize_delta(real_seen_delta, syn_unseen_delta, real_unseen_delta, scale=1000, save_path='hhh'):
    """协方差矩阵可视化。可视化随机三个已见类，和三个未见类(合成/真实)"""
    rsd = real_seen_delta[:, :10, :10].cpu().numpy()    
    sud = syn_unseen_delta[:, :10, :10].cpu().numpy()
    rud = real_unseen_delta[:, :10, :10].cpu().numpy()

    ids = np.random.permutation(500)[:3]
    rsd = rsd[ids] / scale  # for seen
    sud = sud[ids] / scale  # for unseen
    rud = rud[ids] / scale

    plt.figure(figsize=(24, 27))
    
    for k in range(ids.shape[0]): # 遍历类
        # real seen
        plt.subplot(3, 3, 1+3*k)
        plt.title('Seen')
        plt.imshow(rsd[k], cmap='viridis', interpolation='nearest')
        for i in range(rsd[k].shape[0]):
            for j in range(rsd[k].shape[1]):
                plt.text(j, i, f'{rsd[k][i, j]:.2f}', ha='center', va='center', color='w')    

        # syn unseen
        plt.subplot(3, 3, 2+3*k)
        plt.title('Unseen (Syn)')
        plt.imshow(sud[k], cmap='viridis', interpolation='nearest')
        for i in range(sud[k].shape[0]):
            for j in range(sud[k].shape[1]):
                plt.text(j, i, f'{sud[k][i, j]:.2f}', ha='center', va='center', color='w')    
        
        # real unseen
        plt.subplot(3, 3, 3+3*k)
        plt.title('Unseen (Real)')
        plt.imshow(rud[k], cmap='viridis', interpolation='nearest')
        for i in range(rud[k].shape[0]):
            for j in range(rud[k].shape[1]):
                plt.text(j, i, f'{rud[k][i, j]:.2f}', ha='center', va='center', color='w')    

    # 显示图形
    plt.savefig(save_path)
    return


def extract_feature_and_visualization(testset, model, opt, logger):
    """可视化测试集中seen和unseen的特征和印刷原型"""
    model.eval()
    with torch.no_grad():
        # prn prototype
        seen_proto = model.feat_extractor(model.printed_charimgs[model.y_trn]).cpu().detach().numpy()
        unseen_proto = model.feat_extractor(model.printed_charimgs[model.y_tst]).cpu().detach().numpy()
        # seen feat
        tst_seen_feat = []
        for i in tqdm(range(model.y_trn.shape[0])):
            feat = model.feat_extractor(testset.test_seen[i].cuda()).cpu().detach().numpy()
            tst_seen_feat.append(feat)
        # unseen feat
        tst_unseen_feat = []
        for i in tqdm(range(model.y_tst.shape[0])):
            feat = model.feat_extractor(testset.test_unseen[i].cuda()).cpu().detach().numpy()
            tst_unseen_feat.append(feat)

    assert seen_proto.shape[1] == 2

    # mean shift
    seen_mean_shift = []
    for i in tqdm(range(model.y_trn.shape[0])):
        d = np.linalg.norm(tst_seen_feat[i].mean(0) - seen_proto[i], 2)
        seen_mean_shift.append(d)
    unseen_mean_shift = []
    for i in tqdm(range(model.y_tst.shape[0])):
        d = np.linalg.norm(tst_unseen_feat[i].mean(0) - unseen_proto[i], 2)
        unseen_mean_shift.append(d)
    logger.info('Proto-to-Mean shift')
    logger.info('seen: ')
    logger.info(seen_mean_shift)
    logger.info('unseen: ')
    logger.info(unseen_mean_shift)
    
    # Covariance
    seen_eigenvalues = []
    for i in tqdm(range(model.y_trn.shape[0])):
        zero_mean = tst_seen_feat[i] - tst_seen_feat[i].mean(0, keepdims=True)
        cov = zero_mean.T @ zero_mean / tst_seen_feat[i].shape[0]      
        v, _ = np.linalg.eig(cov)
        seen_eigenvalues.append(list(np.sqrt(v)))
    unseen_eigenvalues = []
    for i in tqdm(range(model.y_tst.shape[0])):
        zero_mean = tst_unseen_feat[i] - tst_unseen_feat[i].mean(0, keepdims=True)
        cov = zero_mean.T @ zero_mean / tst_unseen_feat[i].shape[0]      
        v, _ = np.linalg.eig(cov)
        unseen_eigenvalues.append(list(np.sqrt(v)))
    logger.info('The Eigenvalues')
    logger.info('seen: ')
    logger.info(seen_eigenvalues)
    logger.info('unseen: ')
    logger.info(unseen_eigenvalues)

    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(20, 5))
    # colors = tuple([(np.random.random(),np.random.random(), np.random.random()) 
    #                     for _ in range(model.y_trn.shape[0] + model.y_tst.shape[0])])
    # colors = [rgb2hex(x) for x in colors]
    colors = plt.cm.tab20.colors
    # seen
    for i in tqdm(range(model.y_trn.shape[0])):
        axs[0].scatter([seen_proto[i][0]], [seen_proto[i][1]], color=colors[i], marker='*', s=80, edgecolors='k', linewidths=0.5)
        axs[0].scatter(tst_seen_feat[i][:, 0], tst_seen_feat[i][:, 1], color=colors[i], label=i, marker='.', s=15)
    axs[0].axis('equal')
    axs[0].set_title('Seen')

    # unseen. Just 5 not 10 classes for clearness
    N = model.y_trn.shape[0]
    # for i in tqdm(range(model.y_tst.shape[0])):
    for i in tqdm(range(5)):
        axs[1].scatter([unseen_proto[i][0]], [unseen_proto[i][1]], color=colors[i+N], marker='*', s=80, edgecolors='k', linewidths=0.5)
        axs[1].scatter(tst_unseen_feat[i][:, 0], tst_unseen_feat[i][:, 1], color=colors[i+N], label=i+N, marker='x', s=10)
    axs[1].axis('equal')
    axs[1].set_title('Unseen')

    # seen + unseen
    for i in tqdm(range(model.y_trn.shape[0])):
        axs[2].scatter([seen_proto[i][0]], [seen_proto[i][1]], color=colors[i], marker='*', s=80, edgecolors='k', linewidths=0.5)
        axs[2].scatter(tst_seen_feat[i][:, 0], tst_seen_feat[i][:, 1], color=colors[i], label=i, marker='.', s=15)
    N = model.y_trn.shape[0]
    # for i in tqdm(range(model.y_tst.shape[0])):
    for i in tqdm(range(5)):
        axs[2].scatter([unseen_proto[i][0]], [unseen_proto[i][1]], color=colors[i+N], marker='*', s=80, edgecolors='k', linewidths=0.5)
        axs[2].scatter(tst_unseen_feat[i][:, 0], tst_unseen_feat[i][:, 1], color=colors[i+N], label=i+N, marker='x', s=10)
    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[2].axis('equal')
    axs[2].set_title('All')

    plt.savefig(opt.outf+'/vis_feat.pdf')
