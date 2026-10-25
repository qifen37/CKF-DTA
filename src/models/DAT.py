# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from src.models.encoders import (
    SMILESTransformerEncoder, DrugMGCN,
    ProteinSeqEncoder, ProteinGraphEncoder,
    KGEncoder
)
from src.models.fusion import FusionAttention, InfoNCELoss, PredictionModule


class CKFDTA(nn.Module):
    def __init__(self, config=None):
        super(CKFDTA, self).__init__()

        if config is None:
            config = {}

        self.drug_2d = DrugMGCN(
            num_feature_xd=config.get('num_feature_xd', 78),
            output_dim=config.get('drug_graph_dim', 128),
            dropout=config.get('dropout', 0.2)
        )

        self.prot_1d = ProteinSeqEncoder(
            esm_dim=config.get('esm_dim', 1280),
            rnn_hidden=config.get('rnn_hidden', 256),
            rnn_layers=config.get('rnn_layers', 2),
            output_dim=config.get('prot_seq_dim', 512),
            dropout=config.get('dropout', 0.2)
        )

        self.prot_2d = ProteinGraphEncoder(
            input_dim=config.get('esm_dim', 1280),
            hidden_dim=config.get('prot_graph_hidden', 128),
            num_layers=config.get('prot_graph_layers', 5),
            output_dim=config.get('prot_graph_dim', 128),
            dropout=config.get('dropout', 0.2)
        )

        self.drug_1d = SMILESTransformerEncoder(
            vocab_size=config.get('smiles_vocab_size', 63),
            max_len=config.get('max_smiles_len', 85),
            d_model=config.get('transformer_d_model', 128),
            nhead=config.get('transformer_nhead', 8),
            num_layers=config.get('transformer_layers', 2),
            dim_feedforward=config.get('transformer_ff_dim', 512),
            output_dim=config.get('drug_seq_dim', 512),
            dropout=config.get('dropout', 0.1)
        )

        self.drug_kg = KGEncoder(
            input_dim=config.get('kg_emb_dim', 400),
            output_dim=config.get('kg_output_dim', 512)
        )

        self.prot_kg = KGEncoder(
            input_dim=config.get('kg_emb_dim', 400),
            output_dim=config.get('kg_output_dim', 512)
        )

        self.drug_fusion = FusionAttention(
            seq_dim=config.get('drug_seq_dim', 512),
            graph_dim=config.get('drug_graph_dim', 128),
            kg_dim=config.get('kg_output_dim', 512),
            fused_dim=config.get('fused_dim', 512),
            nhead=config.get('fusion_nhead', 8),
            dropout=config.get('dropout', 0.1)
        )

        self.prot_fusion = FusionAttention(
            seq_dim=config.get('prot_seq_dim', 512),
            graph_dim=config.get('prot_graph_dim', 128),
            kg_dim=config.get('kg_output_dim', 512),
            fused_dim=config.get('fused_dim', 512),
            nhead=config.get('fusion_nhead', 8),
            dropout=config.get('dropout', 0.1)
        )

        self.cl_loss_fn = InfoNCELoss(
            temperature=config.get('temperature', 0.07),
            projection_dim=config.get('cl_projection_dim', 256),
            seq_dim=config.get('drug_seq_dim', 512),
            graph_dim=config.get('drug_graph_dim', 128),
            sg_dim=config.get('fused_dim', 512),
            kg_dim=config.get('kg_output_dim', 512)
        )

        self.predictor = PredictionModule(
            drug_fused_dim=config.get('fused_dim', 512),
            prot_fused_dim=config.get('fused_dim', 512),
            hidden_dims=config.get('predictor_hidden_dims', [512, 256]),
            dropout=config.get('dropout', 0.2),
            use_attention=config.get('use_prediction_attention', True)
        )

    def forward(self, smiles_tokens, drug_x, drug_edge_index, drug_batch,
                prot_esm, prot_x, prot_edge_index, prot_batch,
                drug_kg_emb, prot_kg_emb):

        f_seq_d = self.drug_1d(smiles_tokens)

        f_graph_d = self.drug_2d(drug_x, drug_edge_index, drug_batch)

        f_seq_p = self.prot_1d(prot_esm)

        f_graph_p = self.prot_2d(prot_x, prot_edge_index, prot_batch)

        h_kg_d = self.drug_kg(drug_kg_emb)

        h_kg_p = self.prot_kg(prot_kg_emb)

        f_fused_d, f_sg_d = self.drug_fusion(f_seq_d, f_graph_d, h_kg_d)
        f_fused_p, f_sg_p = self.prot_fusion(f_seq_p, f_graph_p, h_kg_p)

        y_pred = self.predictor(f_fused_d, f_fused_p)

        cl_loss = self.cl_loss_fn(
            f_seq_d, f_graph_d, f_sg_d, h_kg_d,
            f_seq_p, f_graph_p, f_sg_p, h_kg_p
        )

        return y_pred, cl_loss
