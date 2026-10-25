# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch_sparse
from torch_geometric.nn import GCNConv, GINConv, global_max_pool as gmp, global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_mean


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=85, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SMILESTransformerEncoder(nn.Module):
    def __init__(self, vocab_size=63, max_len=85, d_model=128, nhead=8,
                 num_layers=2, dim_feedforward=512, output_dim=512, dropout=0.1):
        super(SMILESTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.pad_idx = vocab_size

        self.token_embedding = nn.Embedding(
            vocab_size + 1, d_model, padding_idx=self.pad_idx
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, smiles_tokens):
        src_key_padding_mask = (smiles_tokens == self.pad_idx) 

        x = self.token_embedding(smiles_tokens) * math.sqrt(self.d_model) 
        x = self.pos_encoder(x)

        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )
        mask = ~src_key_padding_mask
        mask = mask.unsqueeze(-1)
        x = x.masked_fill(~mask, float('-inf'))
        x = x.max(dim=1)[0]

        x = self.fc_out(x)
        return x

class DrugMGCN(nn.Module):
    def __init__(self, num_feature_xd=78, output_dim=128, dropout=0.2):
        super(DrugMGCN, self).__init__()

        self.conv1_1 = GCNConv(num_feature_xd, num_feature_xd)
        self.conv1_2 = GCNConv(num_feature_xd, num_feature_xd * 2)
        self.conv1_3 = GCNConv(num_feature_xd * 2, num_feature_xd * 4)

        self.conv2_1 = GCNConv(num_feature_xd, num_feature_xd)
        self.conv2_2 = GCNConv(num_feature_xd, num_feature_xd * 2)

        self.conv3_1 = GCNConv(num_feature_xd, num_feature_xd)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        concat_dim = num_feature_xd * 4 + num_feature_xd * 2 + num_feature_xd
        self.fc_g1 = nn.Linear(concat_dim, 512)
        self.fc_g2 = nn.Linear(512, output_dim)

    def forward(self, x, edge_index, batch):
        adj = to_dense_adj(edge_index)
        num_nodes = adj.shape[1]

        edge_index_sq, _ = torch_sparse.spspmm(
            edge_index, None, edge_index, None,
            num_nodes, num_nodes, num_nodes, coalesced=True
        )
        edge_index_cb, _ = torch_sparse.spspmm(
            edge_index_sq, None, edge_index, None,
            num_nodes, num_nodes, num_nodes, coalesced=True
        )

        h1 = self.relu(self.conv1_1(x, edge_index))
        h1 = self.relu(self.conv1_2(h1, edge_index))
        h1 = self.relu(self.conv1_3(h1, edge_index))

        h2 = self.relu(self.conv2_1(x, edge_index_sq))
        h2 = self.relu(self.conv2_2(h2, edge_index_sq))

        h3 = self.relu(self.conv3_1(x, edge_index_cb))

        concat = torch.cat([h1, h2, h3], dim=1)
        xd = gmp(concat, batch)

        xd = self.relu(self.fc_g1(xd))
        xd = self.dropout(xd)
        xd = self.fc_g2(xd)
        xd = self.dropout(xd)

        return xd


class ProteinSeqEncoder(nn.Module):
    def __init__(self, esm_dim=1280, rnn_hidden=256, rnn_layers=2,
                 output_dim=512, dropout=0.2):
        super(ProteinSeqEncoder, self).__init__()

        self.input_fc = nn.Linear(esm_dim, rnn_hidden)

        self.bilstm = nn.LSTM(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc_out = nn.Linear(rnn_hidden * 2, output_dim)

    def forward(self, prot_esm):
        x = self.input_fc(prot_esm)
        x, _ = self.bilstm(x)
        x = x.max(dim=1)[0]
        x = self.fc_out(x)
        return x


class VirtualNode(nn.Module):
    def __init__(self, hidden_dim):
        super(VirtualNode, self).__init__()
        self.vn_embedding = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.vn_embedding.weight.data, 0)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1
        vn = self.vn_embedding(torch.zeros(batch_size, dtype=torch.long, device=x.device))

        x = x + vn[batch]

        vn_agg = scatter_mean(x, batch, dim=0, dim_size=batch_size)

        vn = vn + self.mlp(vn_agg)

        return x, vn


class ProteinGraphEncoder(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=128, num_layers=5,
                 output_dim=128, dropout=0.2):
        super(ProteinGraphEncoder, self).__init__()

        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gin_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        self.virtual_node = VirtualNode(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)

        for i in range(self.num_layers):
            h = self.gin_layers[i](x, edge_index)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout(h)
            x = x + h

        x, vn = self.virtual_node(x, batch)

        x = global_mean_pool(x, batch)

        x = self.fc_out(x)
        return x


class KGEncoder(nn.Module):
    def __init__(self, input_dim=400, output_dim=512):
        super(KGEncoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, kg_emb):
        x = kg_emb.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x).squeeze(-1)

        x = self.fc(x)
        return x
