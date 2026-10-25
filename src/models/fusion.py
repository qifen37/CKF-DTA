# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % nhead == 0, f"d_model({d_model}) must be divisible by nhead({nhead})"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b):
        B = x_a.size(0)

        Q = self.W_q(x_a).view(B, 1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(x_b).view(B, 1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x_b).view(B, 1, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, self.d_model)

        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + x_a)

        return output


class FusionAttention(nn.Module):
    def __init__(self, seq_dim=512, graph_dim=128, kg_dim=512,
                 fused_dim=512, nhead=8, dropout=0.1):
        super(FusionAttention, self).__init__()

        self.seq_proj = nn.Linear(seq_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.kg_proj = nn.Linear(kg_dim, fused_dim)

        self.cross_attn_sg = MultiHeadCrossAttention(
            d_model=fused_dim, nhead=nhead, dropout=dropout
        )

        self.cross_attn_kg = MultiHeadCrossAttention(
            d_model=fused_dim, nhead=nhead, dropout=dropout
        )

    def forward(self, f_seq, f_graph, h_kg):
        f_seq_proj = self.seq_proj(f_seq)
        f_graph_proj = self.graph_proj(f_graph)
        h_kg_proj = self.kg_proj(h_kg)

        f_sg = self.cross_attn_sg(f_seq_proj, f_graph_proj)

        f_fused = self.cross_attn_kg(f_sg, h_kg_proj)

        return f_fused, f_sg


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, projection_dim=256,
                 seq_dim=512, graph_dim=128, sg_dim=512, kg_dim=512):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

        self.proj_seq = nn.Sequential(
            nn.Linear(seq_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.proj_graph = nn.Sequential(
            nn.Linear(graph_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.proj_sg = nn.Sequential(
            nn.Linear(sg_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.proj_kg = nn.Sequential(
            nn.Linear(kg_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def _infonce(self, z_a, z_b):
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        logits = torch.matmul(z_a, z_b.T) / self.temperature

        B = z_a.size(0)
        labels = torch.arange(B, device=z_a.device)

        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)

        return (loss_ab + loss_ba) / 2.0

    def forward(self, f_seq_d, f_graph_d, f_sg_d, h_kg_d,
                f_seq_p, f_graph_p, f_sg_p, h_kg_p):
        z_seq_d = self.proj_seq(f_seq_d)
        z_graph_d = self.proj_graph(f_graph_d)
        loss_sg_d = self._infonce(z_seq_d, z_graph_d)

        z_seq_p = self.proj_seq(f_seq_p)
        z_graph_p = self.proj_graph(f_graph_p)
        loss_sg_p = self._infonce(z_seq_p, z_graph_p)

        z_sg_d = self.proj_sg(f_sg_d)
        z_kg_d = self.proj_kg(h_kg_d)
        loss_kg_d = self._infonce(z_sg_d, z_kg_d)

        z_sg_p = self.proj_sg(f_sg_p)
        z_kg_p = self.proj_kg(h_kg_p)
        loss_kg_p = self._infonce(z_sg_p, z_kg_p)

        cl_loss = (loss_sg_d + loss_sg_p + loss_kg_d + loss_kg_p) / 4.0

        return cl_loss


class MultiHeadLinearAttention(nn.Module):
    def __init__(self, input_dim=512, n_heads=4, dropout=0.1):
        super(MultiHeadLinearAttention, self).__init__()
        assert input_dim % n_heads == 0, \
            f"input_dim({input_dim}) must be divisible by n_heads({n_heads})"

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)

        self.out_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        B = x.size(0)

        Q = self.W_q(x).view(B, self.n_heads, self.head_dim)
        K = self.W_k(x).view(B, self.n_heads, self.head_dim)

        attn_scores = (Q * K) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        x_reshaped = x.view(B, self.n_heads, self.head_dim)
        out = attn_weights * x_reshaped

        out = out.contiguous().view(B, self.input_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        out = self.layer_norm(out + x)
        return out


class PredictionModule(nn.Module):
    def __init__(self, drug_fused_dim=512, prot_fused_dim=512,
                 hidden_dims=None, dropout=0.2, use_attention=True):
        super(PredictionModule, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        input_dim = drug_fused_dim + prot_fused_dim

        self.use_attention = use_attention
        if use_attention:
            self.drug_attn = MultiHeadLinearAttention(drug_fused_dim, n_heads=4)
            self.prot_attn = MultiHeadLinearAttention(prot_fused_dim, n_heads=4)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, f_fused_drug, f_fused_prot):
        if self.use_attention:
            f_fused_drug = self.drug_attn(f_fused_drug)
            f_fused_prot = self.prot_attn(f_fused_prot)

        xc = torch.cat([f_fused_drug, f_fused_prot], dim=1)

        pred = self.dnn(xc).squeeze(-1)

        return pred
