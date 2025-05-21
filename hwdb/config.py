import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam. default=0.9')
parser.add_argument('--workers', type=int, default=64)
parser.add_argument('--mode', default='train', help='train|val|test')
parser.add_argument('--epochs', type=int, default=30, help='the total number of training epoch')
parser.add_argument('--trainBatchSize', type=int, default=256, help='the mini-batch size of training')
parser.add_argument('--testBatchSize', type=int, default=256, help='the mini-batch size of test')
parser.add_argument('--labelDim', type=int, default=272, help='the dim of character label')  # HDE embedding
parser.add_argument('--saveFilename', default='checkpoint')
parser.add_argument('--negativeSize', type=int, default=20)
parser.add_argument('--nodeSize', type=int, default=7643)
parser.add_argument('--neighbor_k', type=int, default=3)
parser.add_argument('--pairwise', default='root', help='tree|root')
parser.add_argument('--dataFilename', default='HWDB_subset_3755.hdf5')
parser.add_argument('--test', action='store_true', help='true|false')
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--checkpoint', type=str, default='shit', help='the checkpoint for test')
parser.add_argument('--feaDim', type=int, default=512, help='the dim of features')  
parser.add_argument('--backbone', type=str, default='resnetlike', help='resnetlike|resnetlike-v2|resnetlike-v3|resnetlike-v4|')  
parser.add_argument('--input_size', type=int, default=64, help='The input image size')


parser.add_argument('--gpu_id', type=int, default=5, help='GPU id')
parser.add_argument('--outf', default='1222_01_500', help='the name of output dir')
parser.add_argument('--task', type=int, default=1, help='0|1|2|3|4')                      # 0:2755-1000 1:2755-1000(random) 2: 3755-3755, 3: 3755-3008, 4: 20%-80%
parser.add_argument('--seenSize', type=int, default=500, help='500|1000|1500|2000|2755')  # 500, 1000, 1500, 2000, 2755 (add it by myself!)
parser.add_argument('--gzsl', action='store_true', help='generalized ZSL')

parser.add_argument('--rate', type=float, default=0.6, help='.1|.2|.4|.6|.8|.9')
parser.add_argument('--randomseed', type=int, default=2023)
parser.add_argument('--loss', type=int, default=3, help='0|1|2|3')

# printed          
parser.add_argument('--samplingSize', type=int, default=500, help='the sampling size of printed images in a training batch')
parser.add_argument('--distance_prn', type=str, default='euclidean', help='cosine|euclidean|innerdot|euclidean_square')
parser.add_argument('--scale_choice', type=str, default='learnable', help='the type of scale in DCE. constant | learnable')
parser.add_argument('--scale_weight', type=float, default=1.0, help='the scale value in DCE') 
parser.add_argument('--lambda_pl', type=float, default=0.000, help='the weight of pl loss') 

# synthesized data
parser.add_argument('--syn_path', default='peace_and_love', help='the path to sythesized data')
parser.add_argument('--syn_kshot', type=int, default=100, help='k shot in synthesized data')

# Bayes Classifier
parser.add_argument('--bayes', action='store_true', help='whether use the bayes classifier in gaussion assumption')
parser.add_argument('--cal_mu', type=str, default='intp', choices=['intp', 'no', 'raw', 'ideal'], help='how to calculate the mean of unseen')
parser.add_argument('--cal_cov', type=str, default='RDA', choices=[ 'no', 'QDF', 'LSQDF', 'LSQDF-S', 'LSQDF-A', 'LDF', 'LDF-S', 'LDF-A', 'QDF+LDF', 'QDF+LDF-S', 'QDF+LDF-A', 'RDA', 'RDA-S', 'RDA-A', 'ideal'], help='how to calculate the covariance of unseen')
parser.add_argument('--mqdf', action='store_true', help='whether use the MQDF')

parser.add_argument('--interpolation1', type=float, default=0.5, help='the hyparparameter in mean calibration') 
parser.add_argument('--alpha', type=float, default=0.8, help='the hyparparameter in cov calibration') 
parser.add_argument('--beta', type=float, default=0.2, help='the hyparparameter in cov calibration') 


opt = parser.parse_args()
opt.cuda = True