from __future__ import print_function,division

import os
import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import re
import csv
import random
import copy
from math import sqrt
from scipy import stats
import torch.nn.functional as F
from process_smiles import *
import time
device_ids = [0, 1, 2, 3]

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def kd_loss(guide, hint, outputs, labels, alpha=0.2):
        KD_loss = F.mse_loss(guide, hint) * \
            alpha + F.mse_loss(outputs, labels) * (1. - alpha)

        return KD_loss

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs



def rm_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r2 = r_squared_error(y_true, y_pred)

    k = np.sum(y_true * y_pred) / (np.sum(y_pred ** 2) + 1e-10)
    y_pred_0 = k * y_pred
    ss_res = np.sum((y_true - y_pred_0) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10
    r02 = 1 - ss_res / ss_tot

    rm2 = r2 * (1 - np.sqrt(np.abs(r2 - r02)))
    return rm2


def pack_sequences(X, lengths, padding_idx, order=None):
    
    #X = [x.squeeze(0) for x in X]
    
    n = len(X)
    #lengths = np.array([len(x) for x in X])
    if order is None:
        order = np.argsort(lengths)[::-1]
    m = max(lengths)
    
    X_block = X[0].new(n,m).zero_() + padding_idx
    
    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i,:len(x)] = x

    return X_block.cuda(), order

def pack_pre_sequences(X, lengths, padding_idx=0, order=None):
    n = len(X)
    hidden_size = X[0].shape[1]
    if order is None:
        order = np.argsort(lengths)[::-1]
    m = max(lengths)
    X_block = X[0].new(n,m,hidden_size).zero_() + padding_idx
    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i,:len(x),:] = x
    
    return X_block, order

def unpack_sequences(X, order):
    X,lengths = pad_packed_sequence(X, batch_first=True)
    X_block = torch.zeros(size=X.size())
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i]
    return X_block.cuda()

def split_text(text, length):
    text_arr = re.findall(r'.{%d}' % int(length), text)
    return(text_arr)  # ['123', '456', '789', 'abc', 'def']


def load_protvec(filename):
	protvec = []
	key_aa = {}
	count = 0
	with open(filename, "r") as csvfile:
		protvec_reader = csv.reader(csvfile, delimiter='\t')
		for k, row in enumerate(protvec_reader):
			if k == 0:
				continue
			protvec.append([float(x) for x in row[1:]])
			key_aa[row[0]] = count
			count = count + 1

	protvec.append([0.0] * 100)
	key_aa["zero"] = count
	return protvec, key_aa

class DrugTargetDataset(torch.utils.data.Dataset):
    def __init__(self, X0, X1, Y, pid, is_target_pretrain=True, is_drug_pretrain=False, self_link=True,
                 dataset='davis', kg_emb_path=None):
        self.X0 = X0  # drug SMILES
        self.X1 = X1  # protein sequences
        self.Y = Y
        self.pid = pid
        self.smilebet = Smiles()

        smiles = copy.deepcopy(self.X0)
        self.node_drugs = []
        self.edge_drugs = []
        for smile in smiles:
            node_drug, edge_drug = smile_to_graph(smile)
            self.node_drugs.append(node_drug)
            self.edge_drugs.append(edge_drug)
        self.node_counts = [node_drug[0] for node_drug in self.node_drugs]

        self.smiles_tokens = [encode_smiles(s) for s in self.X0]

        self.is_target_pretrain = is_target_pretrain
        self.is_drug_pretrain = is_drug_pretrain

        z = np.load(str(dataset) + '.npz', allow_pickle=True)
        self.z = z['dict'][()]

        node_proteins = np.load('data/node/' + str(dataset) + '.npz', allow_pickle=True)
        self.node_proteins = node_proteins['dict'][()]
        edge_proteins = np.load('data/edge/' + str(dataset) + '.npz', allow_pickle=True)
        self.edge_proteins = edge_proteins['dict'][()]


        if kg_emb_path is not None and os.path.exists(kg_emb_path):
            kg_data = np.load(kg_emb_path, allow_pickle=True)
            self.drug_kg_emb = kg_data['drug']      # (num_drugs, 400)
            self.prot_kg_emb = kg_data['protein']    # (num_proteins, 400)
        else:
            self.drug_kg_emb = None
            self.prot_kg_emb = None

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        prot = torch.from_numpy(self.z[self.pid[i]]).squeeze()
        node_drugs = torch.tensor(np.array(self.node_drugs[i]), dtype=torch.float)
        edge_drugs = torch.tensor(np.array(self.edge_drugs[i]), dtype=torch.long).transpose(1, 0)
        node_proteins = self.node_proteins[self.pid[i]].squeeze()
        edge_proteins = torch.tensor(self.edge_proteins[self.pid[i]], dtype=torch.float32).squeeze()

        smiles_tok = self.smiles_tokens[i]

        if self.drug_kg_emb is not None:
            drug_kg = torch.tensor(self.drug_kg_emb[i % len(self.drug_kg_emb)], dtype=torch.float)
            prot_kg = torch.tensor(self.prot_kg_emb[self.pid[i] % len(self.prot_kg_emb)], dtype=torch.float)
        else:
            drug_kg = torch.zeros(400, dtype=torch.float)
            prot_kg = torch.zeros(400, dtype=torch.float)

        return [prot, smiles_tok, self.Y[i], node_drugs, edge_drugs,
                node_proteins, edge_proteins, drug_kg, prot_kg]

def collate(args):
    prot_esm_list = [a[0] for a in args]
    smiles_tokens = [a[1] for a in args]
    y = [a[2] for a in args]
    node_drugs = [a[3] for a in args]
    edge_drugs = [a[4] for a in args]
    node_proteins_list = [a[5] for a in args]
    edge_proteins_list = [a[6] for a in args]
    drug_kg_list = [a[7] for a in args]
    prot_kg_list = [a[8] for a in args]

    drug_batch = []
    cumulative_nodes = 0
    for idx, nodes in enumerate(node_drugs):
        drug_batch.extend([idx] * nodes.size(0))
        if idx > 0:
            edge_drugs[idx] += cumulative_nodes
        cumulative_nodes += nodes.size(0)

    prot_nodes_all = []
    prot_edges_all = []
    prot_batch_list = []
    cumulative_prot_nodes = 0
    for idx, (node_feat, adj) in enumerate(zip(node_proteins_list, edge_proteins_list)):
        if isinstance(node_feat, np.ndarray):
            node_feat = torch.tensor(node_feat, dtype=torch.float)
        if isinstance(adj, np.ndarray):
            adj = torch.tensor(adj, dtype=torch.float32)

        num_nodes = node_feat.size(0)
        prot_nodes_all.append(node_feat)
        prot_batch_list.extend([idx] * num_nodes)

        edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()
        if edge_index.size(1) > 0:
            edge_index = edge_index + cumulative_prot_nodes
        prot_edges_all.append(edge_index)

        cumulative_prot_nodes += num_nodes

    y_stacked = torch.stack(y, 0)
    smiles_tokens_stacked = torch.stack(smiles_tokens, 0)
    node_drugs_concat = torch.cat(node_drugs, dim=0)
    edge_drugs_concat = torch.cat(edge_drugs, dim=1)
    drug_batch_tensor = torch.tensor(drug_batch, dtype=torch.long)

    prot_esm_stacked = torch.stack(prot_esm_list, 0)

    prot_nodes_concat = torch.cat(prot_nodes_all, dim=0) if prot_nodes_all else torch.zeros(0, 1280)
    prot_edges_concat = torch.cat(prot_edges_all, dim=1) if prot_edges_all else torch.zeros(2, 0, dtype=torch.long)
    prot_batch_tensor = torch.tensor(prot_batch_list, dtype=torch.long)

    drug_kg_stacked = torch.stack(drug_kg_list, 0)
    prot_kg_stacked = torch.stack(prot_kg_list, 0)

    return (prot_esm_stacked, smiles_tokens_stacked, y_stacked,
            node_drugs_concat, edge_drugs_concat, drug_batch_tensor,
            prot_nodes_concat, prot_edges_concat, prot_batch_tensor,
            drug_kg_stacked, prot_kg_stacked)


class Alphabets():
    def __init__(self, chars, encoding=None, missing=255):
        self.chars = np.frombuffer(chars, dtype='uint8')
        self.size = len(self.chars)
        self.encoding = np.zeros(256, dtype='uint8') + missing
        if encoding == None:
            self.encoding[self.chars] = np.arange(self.size)
        else:
            self.encoding[self.chars] = encoding
            
    def encode(self, s):
        s = np.frombuffer(s, dtype='uint8')
        return self.encoding[s]
    
class AminoAcid(Alphabets):
    def __init__(self):
        chars = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        super(AminoAcid, self).__init__(chars)
        
class Smiles(Alphabets):
    def __init__(self):
        chars = b'#%)(+-.1032547698=ACBEDGFIHKMLONPSRUTWVY[Z]_acbedgfihmlonsruty'
        super(Smiles, self).__init__(chars)


MAX_SMILES_LEN = 85
SMILES_PAD_IDX = 63

_smiles_alphabet = Smiles()

def encode_smiles(smiles_str):
    encoded = _smiles_alphabet.encode(smiles_str.encode('utf-8'))
    encoded = encoded[:MAX_SMILES_LEN]
    encoded = np.where(encoded == 255, SMILES_PAD_IDX, encoded)
    padded = np.full(MAX_SMILES_LEN, SMILES_PAD_IDX, dtype=np.int64)
    padded[:len(encoded)] = encoded
    return torch.from_numpy(padded)


def adj_mask(adj, maxsize):
    b = len(adj)
    out = torch.zeros(b, maxsize, maxsize)
    for i in range(b):
        a = adj[i]
        out[i,:a.shape[0],:a.shape[1]] = a
    return out.cuda()

def graph_pad(x, maxsize):
    b = len(x)
    features = x[0].shape[1]
    out = torch.zeros(b, maxsize, features)
    for i in range(b):
        a = x[i]
        out[i,:a.shape[0],:] = a
    return out.cuda(device=a.device)

def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def get_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i != j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
               
    if pair != 0:
        #print(summ)
        #print(pair)
        return summ/pair
    else:
        return 0
    
def feature_mask(sizes, old_edges, rate=0.2):
    # size of feature: [batchsize, num_of_atom, hidden_dim]
    edges = copy.deepcopy(old_edges)
    batchsize = edges.shape[0]
    mask_list = []
    for i in range(batchsize):
        mask_list.append(random.sample(range(sizes[i]),int(sizes[i]*rate)))
        for mask in mask_list[i]:
                edges[i,:,mask] = 0
    #torch.set_printoptions(profile="full")
    return mask_list, edges
