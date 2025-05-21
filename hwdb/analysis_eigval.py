import numpy as np
import matplotlib
matplotlib.rc("font",family='WenQuanYi Micro Hei')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import pickle
from scipy.spatial.distance import pdist, squareform


def plot_eigval(path1, path2):
    ev_s = np.load(path1)
    ev_u = np.load(path2)

    # # 平均
    # ev_s = ev_s.mean(0, keepdims=True)
    # ev_u = ev_u.mean(0, keepdims=True)

    # 开根
    ev_s = np.sqrt(ev_s)
    ev_u = np.sqrt(ev_u)

    # 归一化
    ev_s = ev_s / np.max(ev_s, 1, keepdims=True)  
    ev_u = ev_u / np.max(ev_u, 1, keepdims=True)

    # 截取头部
    ev_s = ev_s[:, :]
    ev_u = ev_u[:, :]
    
    # 随机挑类别
    ns = ev_s.shape[0]
    nu = ev_u.shape[0]
    ind_s = np.random.permutation(ns)[:1]
    ind_u = np.random.permutation(nu)[:1]
    # ind_s = list(range(3))
    # ind_u = list(range(3))

    plt.figure()
    x = list(range(ev_s[0].shape[0]))
    # for i, v in enumerate(ev_s[ind_s]):
    for i, v in enumerate(ev_s[[0]]):
        plt.plot(x, v, label='s%d'%i)   
    # for i, v in enumerate(ev_u[ind_u]):
    for i, v in enumerate(ev_u[[4]]):
        plt.plot(x, v, label='u%d'%i, linestyle='--')   
    plt.legend()
    plt.title('$ \sqrt{\overline{\lambda}(\Sigma)} $', fontsize=20)
    # plt.xlabel('$ i $')
    plt.savefig('hwdb_sqrt_eigval.pdf')
    print('done')


def plot_eigvec(path1, path2):
    """类主成分之间的夹角。
    
    注, 考虑到头部那几个特征值的大小相近, 两个类的主成分夹角应该选其topK成分夹角的最小值。
    """
    n = 10  
    topk = 5

    seen_eig_vec = np.load(path1)
    seen_eig_vec_principle = seen_eig_vec[:n, :, :topk]  
    
    # seen_cos_sim = seen_eig_vec_principle @ seen_eig_vec_principle.T  # T1
    seen_cos_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            seen_cos_sim[i, j] = np.abs(seen_eig_vec_principle[i].T @ seen_eig_vec_principle[j]).max()   # 从topk*topk中选最大的cos
    
    seen_cos_sim = np.clip(np.abs(seen_cos_sim), 0, 1)
    seen_angle = np.degrees(np.arccos(seen_cos_sim))

    unseen_eig_vec = np.load(path2)
    unseen_eig_vec_principle = unseen_eig_vec[:n, :, :topk]
    
    # unseen_cos_sim = unseen_eig_vec_principle @ unseen_eig_vec_principle.T
    unseen_cos_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            unseen_cos_sim[i, j] = np.abs(unseen_eig_vec_principle[i].T @ unseen_eig_vec_principle[j]).max()   # 从topk*topk中选最大的cos

    unseen_cos_sim = np.clip(np.abs(unseen_cos_sim), 0, 1)
    unseen_angle = np.degrees(np.arccos(unseen_cos_sim))  # -> radians -> degrees

    # draw_heatmap(seen_angle, n, title='主成分夹角（度数）', path='real_seen_principle_cos.pdf')
    # draw_heatmap(unseen_angle, n, title='主成分夹角（度数）', path='real_unseen_principle_cos.pdf')
    draw_heatmap(seen_angle, n, title='The angle (in degrees)', path='real_seen_principle_angle.pdf')
    draw_heatmap(unseen_angle, n, title='The angle (in degrees)', path='real_unseen_principle_angle.pdf')
    
    return


def draw_heatmap(data, n, title=None, path=None):    
    # 作图阶段
    fig, ax = plt.subplots()

    # 定义热图的横纵坐标
    xLabel = list(range(n))
    yLabel = list(range(n))

    # 定义横纵坐标的刻度
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    
    # 作图并选择热图的颜色填充风格，这里选择yLGn
    im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=90)
    
    # 增加右侧的颜色刻度条
    plt.colorbar(im)

    # 填充数字
    for i in range(len(yLabel)):
        for j in range(len(xLabel)):
            # print('data[{},{}]:{}'.format(i, j, data[i, j]))
            ax.text(j, i, '%.0f' % data[i, j],
                    ha="center", va="center", color="black")

    # 增加标题
    plt.title(title, fontdict={'size': 18})
    # plt.xlabel('类别', fontdict={'size': 16})
    plt.xlabel('category', fontdict={'size': 16})
    # plt.ylabel('类别', rotation=0, fontdict={'size': 16})
    # plt.ylabel('类', fontdict={'size': 16})
    
    # 保存
    fig.tight_layout()
    plt.savefig(path)
    plt.close()
    return


def plot_eigval_and_feat(seen_feat_path, seen_eigvec_path, seen_eigval_path,
                         unseen_feat_path, unseen_eigvec_path, unseen_eigval_path):
    """为支撑贝叶斯分类器, 同时挑K个已见类和K个未见类来画出(1)协方差的特征值和(2)特征可视化"""
    
    # Loda data
    with open(seen_feat_path, 'rb') as file:
        seen_feat = pickle.load(file)
    seen_eigvec = np.load(seen_eigvec_path)
    seen_eigval = np.load(seen_eigval_path)
    with open(unseen_feat_path, 'rb') as file:
        unseen_feat = pickle.load(file)
    unseen_eigvec = np.load(unseen_eigvec_path)
    unseen_eigval = np.load(unseen_eigval_path)

    # Select K class
    K = 3   # ***********
    seen_class_inds = [0, 50, 100]

    c0 = 0  # 投影向量所在的未见类索引
    nu = len(unseen_feat)
    unseen_mu = [unseen_feat[i].mean(0) for i in range(nu)]
    unseen_mu = np.stack(unseen_mu, 0)
    distance = pdist(unseen_mu, metric='euclidean')
    distance_matrix = squareform(distance)          # inter-class distance
    inds = np.argsort(distance_matrix, axis=-1)     # 升序索引, 由近及远
    # unseen_class_inds = inds[c0][:K]                
    selected = [0, 2, 600]                         # ******挑未见类*******
    unseen_class_inds = inds[c0][selected]

    # Eigenvalue
    ev_s, ev_u = seen_eigval, unseen_eigval
    # 开根
    ev_s = np.sqrt(ev_s)
    ev_u = np.sqrt(ev_u)
    # 归一化
    ev_s = ev_s / np.max(ev_s, 1, keepdims=True)  
    ev_u = ev_u / np.max(ev_u, 1, keepdims=True)
    # # 截取头部
    # ev_s = ev_s[:, :]
    # ev_u = ev_u[:, :]
    # 随机挑类别
    ind_s = seen_class_inds
    ind_u = unseen_class_inds
    
    # Colors
    seen_colors = ['darkorange', 'cyan', 'purple']
    unseen_colors = ['r', 'b', 'g']

    # Plot Eigenvalues
    plt.figure(figsize=(6, 4))
    x = list(range(ev_s[0].shape[0]))
    for i, v in enumerate(ev_s[ind_s]):
        # plt.plot(x, v, color=seen_colors[i], linewidth=2, label='已见类-%d'%i, alpha=0.6)   
        plt.plot(x, v, color=seen_colors[i], linewidth=2, label='Seen-%d'%i, alpha=0.6)   
    for i, v in enumerate(ev_u[ind_u]):
        # plt.plot(x, v, color=unseen_colors[i], linewidth=2, linestyle='--', label='未见类-%d'%i)   
        plt.plot(x, v, color=unseen_colors[i], linewidth=2, linestyle='--', label='Unseen-%d'%i)   
    plt.legend()
    plt.grid()
    plt.title('$ \sqrt{\overline{\lambda}(\Sigma)} $', fontsize=18)
    # plt.savefig('sqrt_eigval.pdf')
    plt.savefig('sqrt_eigval_eng.pdf')
    plt.close()

    # Feature Reduction
    dim_inds = [0, 60]   # ******投影向量的索引******
    proj = unseen_eigvec[c0][:, dim_inds]  
    
    # seen_data = [dict() for _ in range(K)]
    # for i, ind in enumerate(seen_class_inds):
    #     x = seen_feat[ind]  # ?*d
    #     x_proj = x @ proj     # ?*2
    #     x1 = x_proj[:, 0].tolist()
    #     x2 = x_proj[:, 1].tolist()
    #     seen_data[i]['x1'] = x1
    #     seen_data[i]['x2'] = x2
    
    unseen_data = [dict() for _ in range(K)]
    for i, ind in enumerate(unseen_class_inds):
        x = unseen_feat[ind]  # ?*d
        x_proj = x @ proj     # ?*2
        x1 = x_proj[:, 0].tolist()
        x2 = x_proj[:, 1].tolist()
        unseen_data[i]['x1'] = x1
        unseen_data[i]['x2'] = x2

    # Plot Feature
    fig, ax = plt.subplots(figsize=(6, 3))
    # for i in reversed(range(K)):
    #     ax.scatter(seen_data[i]['x1'], seen_data[i]['x2'], color=seen_colors[i], label='已见类-%d'%i)
    for i in reversed(range(K)):
        # ax.scatter(unseen_data[i]['x1'], unseen_data[i]['x2'], color=unseen_colors[i], label='未见类-%d'%i, alpha=0.5)
        ax.scatter(unseen_data[i]['x1'], unseen_data[i]['x2'], color=unseen_colors[i], label='Unseen-%d'%i, alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 1, 0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    # 设置x轴和y轴的尺度一致
    ax.set_aspect('equal')
    # 设置x轴和y轴的刻度间隔相同
    spacing = 500  # 每个刻度间的距离
    ax.xaxis.set_major_locator(ticker.MultipleLocator(spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(spacing))
    plt.xlim(-1500, 2000)
    plt.ylim(-250, 1250)
    plt.tight_layout()
    # plt.savefig('proj_feat.pdf')
    plt.savefig('proj_feat_eng.pdf')
    plt.close()
    return


if __name__ == '__main__':
    # path1 = './results/2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN/real_seen_eigval.npy'
    # path2 = './results/2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN/real_unseen_eigval.npy'
    # plot_eigval(path1, path2)


    path3 = './results/2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN/real_seen_eigvec.npy'
    path4 = './results/2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN/real_unseen_eigvec.npy'
    plot_eigvec(path3, path4)

    # # Data for visualization
    # seen_feat_path = './results/2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN/real_seen_feat.pkl'
    # seen_eigvec_path = './results/2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN/real_seen_eigvec.npy'
    # seen_eigval_path = './results/2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN/real_seen_eigval.npy'
    # unseen_feat_path = './results/2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN/real_unseen_feat.pkl'
    # unseen_eigvec_path = './results/2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN/real_unseen_eigvec.npy'
    # unseen_eigval_path = './results/2024-01-25_01_hwdb_500_CMPL_ndim=256_DSBN/real_unseen_eigval.npy'
    # plot_eigval_and_feat(
    #     seen_feat_path, seen_eigvec_path, seen_eigval_path,
    #     unseen_feat_path, unseen_eigvec_path, unseen_eigval_path,
    # )