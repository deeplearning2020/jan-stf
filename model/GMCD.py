import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv
import math
from torch_geometric.data import Data
from train import DEVICE
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'model'))
from backbone import ResNet50Fc as ResNet50
from utils_HSI import *
import torch.nn.init as init


import numpy as np
import torch
from functools import partial
from sklearn.metrics.pairwise import euclidean_distances
from torch.autograd import Variable
import pdb

def cost_matrix_torch(x, y):
	"Returns the cosine distance"
	# x is the image embedding
	# y is the text embedding
	D = x.size(0)
	x = x.view(D, -1)
	assert(x.size(0)==y.size(0))
	x = x.div(torch.norm(x, p=2, dim=0, keepdim=True) + 1e-12)
	y = y.div(torch.norm(y, p=2, dim=0, keepdim=True) + 1e-12)
	cos_dis = torch.mm(torch.transpose(y,0,1), x)#.t()
	cos_dis = 1 - cos_dis # to minimize this value
	return cos_dis

def IPOT_torch(C, n, m, miu, nu, beta=0.5, device='cuda'):
	# C is the distance matrix
	# c: n by m
	# miu: bs * n
	sigma = torch.ones(int(m), 1).float().to(device)/m # bs * m * 1
	T = torch.ones(n, m).to(device)
	C = torch.exp(-C/beta).float()
	for t in range(20):
		T = C * T # n * m
		for k in range(1):
			delta = miu / torch.squeeze(torch.matmul(T, sigma))
			# a = torch.matmul(torch.transpose(T,0,1), torch.unsqueeze(delta,1))
			# sigma = torch.unsqueeze(nu,1) / a
			sigma = torch.unsqueeze(nu,1) / torch.matmul(torch.transpose(T,0,1), torch.unsqueeze(delta,1))
		# tmp = torch.mm(torch.diag(torch.squeeze(delta)), Q)
		# tmp = torch.unsqueeze(delta,1) * A
		# dim_ = torch.diag(torch.squeeze(sigma)).dim()
		# dim_ = torch.diag(torch.squeeze(sigma)).dim()
		# assert (dim_ == 2 or dim_ == 1, "dim_ is %d" % dim_)
		# T = torch.mm(torch.unsqueeze(delta,1) * T, torch.diag(torch.squeeze(sigma)))
		T = torch.unsqueeze(delta,1) * T * sigma.transpose(1,0)
	return T.detach()

def IPOT_distance_torch(C, n, m, miu, nu, device='cuda'):
	C = C.float().to(device)
	T = IPOT_torch(C, n, m, miu, nu, device=device)
	distance = torch.trace(torch.mm(torch.transpose(C,0,1), T))
	return -distance


def IPOT_distance_torch_batch(C, n, m, miu, nu, iteration, device='cuda'):
	# C as a 2 d matrix
	C = C.float().to(device)
	bs = miu.size(0)
	# if C.dim()==2:
	# 	C=C.repeat(bs, 1, 1)
	if C.dim()==2:
		C = torch.unsqueeze(C, 0)
	# if not bs == C.size(0):
	# 	print('break')
	# assert(bs == C.size(0))
	T = IPOT_torch_batch(C, bs, n, m, miu, nu, iteration, device=device)
	temp = torch.matmul(torch.transpose(C,1,2), T)
	distance = batch_trace(temp, m, bs, device=device)
	return -distance


def IPOT_torch_batch(C, bs, n, m, miu, nu, iteration=20, beta=0.5, device='cuda'):
	# C is the distance matrix, 2d matrix
	# c: n by m
	# miu: bs * n
	sigma = torch.ones(bs, int(m), 1).to(device).detach()/float(m) # bs * m * 1
	Q = torch.ones(bs, n, m).to(device).detach().float()
	C = torch.exp(-C/beta)#.unsqueeze(0)
	if nu.dim() < 3:
		nu = torch.unsqueeze(nu,2)
	# if miu.dim()<3:
	# 	miu = torch.unsqueeze(miu,1)
	miu = torch.squeeze(miu)
	for t in range(iteration):
		Q = C * Q # bs * n * m
		for k in range(1):
			delta = torch.unsqueeze((miu / torch.squeeze(torch.bmm(Q, sigma)+1e-6)),2)
			# delta = ((miu / (torch.bmm(Q, sigma) + 1e-6)))
			a = torch.bmm(torch.transpose(Q,1,2), delta)+1e-6
			sigma = nu / a
		Q = delta * Q * sigma.transpose(2,1)
		# Q = torch.matmul(tmp, diag_sigma)
	return Q.detach()

def IPOT_torch_uniform(C, n, m, beta=0.5, device='cuda'):
	# C is the distance matrix
	sigma = torch.ones(int(m), 1).to(device)/m
	T = torch.ones(n, m).to(device)
	A = torch.exp(-C/beta)
	for t in range(50):
		Q = A * T # n * m
		for k in range(1):
			delta = 1 / (n * torch.mm(Q, sigma))
			a = torch.mm(torch.transpose(Q,0,1), delta)
			sigma = 1 / (float(m) * a)
		tmp = torch.mm(torch.diag(torch.squeeze(delta)), Q)
		dim_ = torch.diag(torch.squeeze(sigma)).dim()
		assert (dim_ == 2 or dim_ == 1)
		T = torch.mm(tmp, torch.diag(torch.squeeze(sigma)))
	return T.detach()

def IPOT_distance_torch_uniform(C, n, m, device='cuda'):
	C = C.float().to(device)
	T = IPOT_torch_uniform(C, n, m, device=device)
	distance = torch.trace(torch.mm(torch.transpose(C,0,1), T))
	return distance


def cost_matrix_batch_torch(x, y):
	"Returns the cosine distance batchwise"
	# x is the image feature: bs * d * m * m
	# y is the audio feature: bs * d * nF
	# return: bs * n * m
	# print(x.size())
	bs = list(x.size())[0]
	D = x.size(1)
	assert(x.size(1)==y.size(1))
	x = x.contiguous().view(bs, D, -1) # bs * d * m^2
	x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
	y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
	cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)#.transpose(1,2)
	cos_dis = 1 - cos_dis # to minimize this value
	# cos_dis = - cos_dis
	return cos_dis.transpose(2,1)


def cost_matrix_batch_torch_acos(x, y):
	"Returns the cosine distance batchwise"
	# x is the image feature: bs * d * m * m
	# y is the audio feature: bs * d * nF
	# return: bs * n * m
	# print(x.size())
	bs = list(x.size())[0]
	D = x.size(1)
	assert(x.size(1)==y.size(1))
	x = x.contiguous().view(bs, D, -1) # bs * d * m^2
	x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
	y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
	cos_dis = torch.bmm(torch.transpose(x,1,2), y)#.transpose(1,2)
	cos_dis = torch.acos(cos_dis) # to minimize this value
	# cos_dis = - cos_dis
	return cos_dis.transpose(2,1)

def cos_batch_torch(x, y):
	"Returns the cosine distance batchwise"
	# x is the image feature: bs * d * m * m
	# y is the audio feature: bs * d * nF
	# return: bs * n * m
	# print(x.size())
	bs = x.size(0)
	D = x.size(1)
	assert(x.size(1)==y.size(1))
	x = x.contiguous().view(bs, D, -1) # bs * d * m^2
	x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
	y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
	cos_dis = torch.bmm(torch.transpose(x,1,2), y)#.transpose(1,2)
	cos_dis = 1 - cos_dis # to minimize this value
	# return cos_dis.transpose(2,1)
	# TODO:
	beta = 0.1
	min_score = cos_dis.min()
	max_score = cos_dis.max()
	threshold = min_score + beta * (max_score - min_score)
	res = cos_dis - threshold
	# res = torch.nn.ReLU()

	return torch.nn.functional.relu(res.transpose(2,1))


def pairwise_distances(x, y=None):
	'''
	Input: x is a Nxd matrix
		   y is an optional Mxd matirx
	Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
			if y is not given then use 'y=x'.
	i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
	'''
	x_norm = (x ** 2).sum(1).view(-1, 1)
	if y is not None:
		y_t = torch.transpose(y, 0, 1)
		y_norm = (y ** 2).sum(1).view(1, -1)
	else:
		y_t = torch.transpose(x, 0, 1)
		y_norm = x_norm.view(1, -1)

	dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
	# Ensure diagonal is zero if x=y
	# if y is None:
	#     dist = dist - torch.diag(dist.diag)
	return torch.clamp(dist, 0.0, np.inf)

def row_pairwise_distances(x, y=None, dist_mat=None):
    if y is None:
        y = x
    if dist_mat is None:
        dtype = x.data.type()
        dist_mat = Variable(torch.Tensor(x.size()[0], y.size()[0]).type(dtype))

    for i, row in enumerate(x.split(1)):
        r_v = row.expand_as(y)
        sq_dist = torch.sum((r_v - y) ** 2, 1)
        dist_mat[i] = sq_dist.view(1, -1)
    return dist_mat

def IPOT_barycenter(p, C, q, iteration=20, beta=0.5, iteration_inner = 1, device='cuda'):
	'''

	:param p: probability vector set, K x n
	:param C: cost matrix, K x n x n
	:param q: initial q, mean of all support, n x d
	:return:
	'''
	K = p.size(0)
	n = p.size(1)
	assert(C.size(1)==C.size(2))
	assert(C.size(1)==p.size(1))
	b = torch.ones(K, int(n), 1).to(device).detach()/float(n) # bs * m * 1
	C = torch.exp(-C/beta)
	T = torch.ones(K, n, n).to(device).detach().float()
	q = torch.unsqueeze(q, 0)
	for t in range(iteration):
		H = T * C
		for k in range(iteration_inner):
			a = q/torch.bmm(H, b)
			b = p/torch.bmm(torch.transpose(H, 2, 1), a)
			q = a * (torch.bmm(H, b))
		T = a * H * b.transpose(2,1)
	return q


def IPOT_distance_torch_batch_uniform(C, bs, n, m, iteration=50, device='cuda'):
	C = C.float().to(device)
	T = IPOT_torch_batch_uniform(C, bs, n, m, iteration=iteration, device=device)
	temp = torch.bmm(torch.transpose(C,1,2), T)
	distance = batch_trace(temp, m, bs, device=device)
	return -distance

def IPOT_distance_torch_batch_uniform_T(C, bs, n, m, iteration=50, device='cuda'):
	C = C.float().to(device)
	T = IPOT_torch_batch_uniform(C, bs, n, m, iteration=iteration, device=device)
	# temp = torch.bmm(torch.transpose(C,1,2), T)
	# distance = batch_trace(temp, m, bs, device=device)
	return T


def IPOT_torch_batch_uniform(C, bs, n, m, beta=0.5, iteration=50, device='cuda'):
	# C is the distance matrix
	# c: bs by n by m
	sigma = torch.ones(bs, int(m), 1).to(device)/float(m)
	T = torch.ones(bs, n, m).to(device)
	A = torch.exp(-C/beta).float().to(device)
	for t in range(iteration):
		Q = A * T # bs * n * m
		for k in range(1):
			delta = 1 / (n * torch.bmm(Q, sigma))
			a = torch.bmm(torch.transpose(Q,1,2), delta)
			sigma = 1 / (float(m) * a)
		T = delta * Q * sigma.transpose(2,1)

	return T#.detach()


def GW_distance(X, Y, p, q, lamda=0.5, iteration=5, OT_iteration=20, device='cuda'):
	'''
	:param X, Y: Source and target embeddings , batchsize by embed_dim by n
	:param p, q: probability vectors
	:param lamda: regularization
	:return: GW distance
	'''
	Cs = cos_batch_torch(X, X).float().to(device)
	Ct = cos_batch_torch(Y, Y).float().to(device)
	# pdb.set_trace()
	bs = Cs.size(0)
	m = Ct.size(2)
	n = Cs.size(2)
	T, Cst = GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration, device=device)
	temp = torch.bmm(torch.transpose(Cst,1,2), T)
	distance = batch_trace(temp, m, bs, device=device)
	return distance

def GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20, device='cuda'):
	one_m = torch.ones(bs, m, 1).float().to(device)
	one_n = torch.ones(bs, n, 1).float().to(device)

	Cst = torch.bmm(torch.bmm(Cs**2, p), torch.transpose(one_m, 1, 2)) + \
	      torch.bmm(one_n, torch.bmm(torch.transpose(q,1,2), torch.transpose(Ct**2, 1, 2))) # bs by n by m
	gamma = torch.bmm(p, q.transpose(2,1)) # outer product, init
	# gamma = torch.einsum('bi,bj->bij', (torch.squeeze(p), torch.squeeze(q))) # outer product, initialization
	for i in range(iteration):
		C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
		# # Sinkhorn iteration
		# b = torch.ones(bs, m, 1).to(device)
		# K = torch.exp(-C_gamma/beta)
		# for i in range(50):cd
		# 	a = p/(torch.bmm(K, b))
		# 	b = q/torch.bmm(K.transpose(1,2), a)
		# gamma = a * K * b
		gamma = IPOT_torch_batch_uniform(C_gamma, bs, n, m, beta=beta, iteration=OT_iteration, device=device)
	Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
	return gamma.detach(), Cgamma

def GW_distance_uniform(X, Y, lamda=1e-1, iteration=5, OT_iteration=20, device='cuda'):
	m = X.size(2)
	n = Y.size(2)
	bs = X.size(0)
	p = (torch.ones(bs, m, 1)/m).to(device)
	q = (torch.ones(bs, n, 1)/n).to(device)
	return GW_distance(X, Y, p, q, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration, device=device)


def batch_diag(a_emb, n, bs, device='cuda'):
	a = torch.eye(n).to(device).unsqueeze(0).repeat(bs, 1, 1) # bs * n * n
	b = (a_emb.unsqueeze(1).repeat(1,n,1))# bs * n * n
	return a*b
	# diagonal bs by n by n

def batch_trace(input_matrix, n, bs, device='cuda'):
	a = torch.eye(n).to(device).unsqueeze(0).repeat(bs, 1, 1)
	b = a * input_matrix
	return torch.sum(torch.sum(b,-1),-1).unsqueeze(1)
	
	

class vgg16(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, init_weights=True, batch_norm=True):
        super(vgg16, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [32, 32, 64, 64],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            'yao': [64, 128, 256, 256]
        }
        layers = []
        for v in cfg['D']:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, stride=1, kernel_size=3)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.LeakyReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.in_channels,
                             self.patch_size, self.patch_size))
            x = self.features(x)
            t, c, w, h = x.size()
        return t * c * w * h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # raw x = (bs, c, ps, ps)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class PredictorCNN(nn.Module):
    def __init__(self, in_dim=1024, num_class=7, prob=0.5, init_type='kaiming_normal'):
        super(PredictorCNN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 4096)
        self.bn1_fc = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 256)
        self.bn2_fc = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_class)
        self.bn_fc3 = nn.BatchNorm1d(num_class)
        self.prob = prob
        self.init_type = init_type
        self._initialize_weights()

    def set_lambda(self, lambd):
        self.lambd = lambd
    
    def _initialize_weights(self):
        if self.init_type == 'kaiming_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'kaiming_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'xavier_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
        elif self.init_type == 'xavier_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x

class PredictorGCN(nn.Module):
    def __init__(self, in_dim,num_class,dropout=0.5, init_type='kaiming_normal'):
        super(PredictorGCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 1024, bias=True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 1024, bias=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(1024, num_class, bias=True)
        self.init_type = init_type
        self._initialize_weights()

    def set_lambda(self, lambd):
        self.lambd = lambd
    
    def _initialize_weights(self):
        if self.init_type == 'kaiming_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'kaiming_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'xavier_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
        elif self.init_type == 'xavier_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

class Topology_Extraction(torch.nn.Module):
    def __init__(self, in_channels):
        super(Topology_Extraction, self).__init__()
        self.conv1 = SAGEConv(in_channels, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = SAGEConv(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        # x_temp_1 = x

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # x_temp_2 = x

        return x

class GraphMCD(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, **kwargs):
        super(GraphMCD, self).__init__()
        self.classes = num_classes
        #@ CNN (backbone / generator / feature extractor)
        self.backbone = vgg16(in_channels, num_classes, patch_size)
        self.backbone_output_dim = self.backbone._get_final_flattened_size()
        #@ GCN
        self.src_gcn = Topology_Extraction(self.backbone_output_dim)
        self.tar_gcn = Topology_Extraction(self.backbone_output_dim)
        self.gcn_output_dim = self.src_gcn.conv2.out_channels
        #@ classifiers
        self.classifierCNN1 = PredictorCNN(in_dim=self.backbone_output_dim, num_class=self.classes, init_type='kaiming_normal')
        # self.classifierCNN2 = PredictorCNN(in_dim=self.backbone_output_dim, num_class=self.classes, init_type='xavier_uniform')
        self.classifierGCN1 = PredictorGCN(in_dim=self.gcn_output_dim, num_class=self.classes, init_type='kaiming_uniform')
        # self.classifierGCN2 = PredictorGCN(in_dim=self.gcn_output_dim, num_class=self.classes, init_type='xavier_normal')

    def forward(self, source, target=None, useOne=False, reverse=False):
        out = self.backbone(source)
        bs = out.shape[0]
        cnn_pred1 = self.classifierCNN1(out, reverse)
        if self.training:
            share_graph = getGraphdataOneDomain(out, bs)
            share_graph = self.src_gcn(share_graph)
            gcn_pred1 = self.classifierGCN1(share_graph, reverse)
            if useOne:
                # return cnn_pred1, gcn_pred1
                return cnn_pred1, gcn_pred1, out, share_graph
            else:
                tar_out = self.backbone(target)
                tar_cnn_pred1 = self.classifierCNN1(tar_out, reverse)
                tar_share_graph = getGraphdataOneDomain(tar_out, bs)
                tar_share_graph = self.tar_gcn(tar_share_graph)
                tar_gcn_pred1 = self.classifierGCN1(tar_share_graph, reverse)
                # return cnn_pred1, gcn_pred1, tar_cnn_pred1, tar_gcn_pred1
                return cnn_pred1, gcn_pred1, tar_cnn_pred1, tar_gcn_pred1,\
                out, share_graph, tar_out, tar_share_graph
        else:
            return cnn_pred1, None

class CrossGraph(nn.Module):
    def __init__(self, in_channels):
        super(CrossGraph, self).__init__()
        self.gcn = Topology_Extraction(in_channels)

    def forward(self, src_feat, tar_feat, src_labels, tar_labels, mask):
        feat = torch.cat((src_feat, tar_feat), dim=0)
        share_graph = getGraphdata_ClassGuided(feat, src_labels, tar_labels, mask)
        if share_graph == None:
            return feat
        share_graph = self.gcn(share_graph)
        return share_graph
