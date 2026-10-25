"""
CKF-DTA Training Script
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from src.getdata import getdata_from_csv
from src.utils import DrugTargetDataset, collate, AminoAcid, ci, mse, rmse, pearson, spearman, rm_squared
from src.models.DAT import CKFDTA


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batchsize', type=int, default=128, help='Number of batch_size')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--pretrain', action='store_false', help='protein pretrained or not')
parser.add_argument('--dataset', default='davis', help='dataset: davis or kiba')
parser.add_argument('--training-dataset-path', default='data/davis_train.csv', help='training dataset path')
parser.add_argument('--testing-dataset-path', default='data/davis_test.csv', help='testing dataset path')

parser.add_argument('--lambda-cl', type=float, default=0.1)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--kg-emb-path', type=str, default=None)
parser.add_argument('--kg-model-path', type=str, default='saved_models/distmult/')

args = parser.parse_args()
dataset = args.dataset
use_cuda = args.cuda and torch.cuda.is_available()

batch_size = args.batchsize
epochs = args.epochs
lr = args.lr
weight_decay = args.weight_decay

is_pretrain = args.pretrain

Alphabet = AminoAcid()

training_dataset_address = args.training_dataset_path
testing_dataset_address = args.testing_dataset_path


if args.kg_emb_path is None:
    kg_emb_path = f'data/kg/{dataset}_kg_embeddings.npz'
else:
    kg_emb_path = args.kg_emb_path

if not os.path.exists(kg_emb_path):
    print(f"KG embedding file {kg_emb_path} not available")


if is_pretrain:
    train_drug, train_protein, train_affinity, pid = getdata_from_csv(training_dataset_address, maxlen=1536)
else:
    train_drug, train_protein, train_affinity = getdata_from_csv(training_dataset_address, maxlen=1024)
    train_protein = [x.encode('utf-8').upper() for x in train_protein]
    train_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in train_protein]
train_affinity = torch.from_numpy(np.array(train_affinity)).float()


dataset_train = DrugTargetDataset(
    train_drug, train_protein, train_affinity, pid,
    is_target_pretrain=is_pretrain, self_link=False,
    dataset=dataset, kg_emb_path=kg_emb_path
)
dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate
)

if is_pretrain:
    test_drug, test_protein, test_affinity, pid = getdata_from_csv(testing_dataset_address, maxlen=1536)
else:
    test_drug, test_protein, test_affinity = getdata_from_csv(testing_dataset_address, maxlen=1024)
    test_protein = [x.encode('utf-8').upper() for x in test_protein]
    test_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in test_protein]
test_affinity = torch.from_numpy(np.array(test_affinity)).float()

dataset_test = DrugTargetDataset(
    test_drug, test_protein, test_affinity, pid,
    is_target_pretrain=is_pretrain, self_link=False,
    dataset=dataset, kg_emb_path=kg_emb_path
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate
)

config = {
    'dropout': args.dropout,
    'temperature': args.temperature,
}
model = CKFDTA(config)

if use_cuda:
    model.cuda()


params = [p for p in model.parameters() if p.requires_grad]
optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

criterion = nn.MSELoss()

train_epoch_size = len(train_drug)
test_epoch_size = len(test_drug)
print('--- KGFusionDTA model --- ')

best_ci = 0
best_mse = 100000

for epoch in range(epochs):
    model.train()
    b = 0
    total_loss = []
    total_ci = []

    for (prot_esm, smiles_tokens, affinity,
         drug_nodes, drug_edges, drug_batch,
         prot_nodes, prot_edge_index, prot_batch,
         drug_kg, prot_kg) in dataloader_train:

        if use_cuda:
            prot_esm = prot_esm.cuda()
            smiles_tokens = smiles_tokens.cuda()
            affinity = affinity.cuda()
            drug_nodes = drug_nodes.cuda()
            drug_edges = drug_edges.cuda()
            drug_batch = drug_batch.cuda()
            prot_nodes = prot_nodes.cuda()
            prot_edge_index = prot_edge_index.cuda()
            prot_batch = prot_batch.cuda()
            drug_kg = drug_kg.cuda()
            prot_kg = prot_kg.cuda()

        y_pred, cl_loss = model(
            smiles_tokens, drug_nodes, drug_edges, drug_batch,
            prot_esm, prot_nodes, prot_edge_index, prot_batch,
            drug_kg, prot_kg
        )

        mse_loss = criterion(y_pred, affinity)
        loss = mse_loss + args.lambda_cl * cl_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        out_cpu = y_pred.cpu().detach()
        affinity_cpu = affinity.cpu().detach()
        loss_cpu = loss.cpu().detach().item()
        c_index = ci(affinity_cpu.numpy(), out_cpu.numpy())

        b = b + batch_size
        total_loss.append(loss_cpu)
        total_ci.append(c_index)
        print('# [{}/{}] training {:.1%} loss={:.5f} (mse={:.5f}, cl={:.5f}), ci={:.5f}'.format(
            epoch + 1, epochs, b / train_epoch_size,
            loss_cpu, mse_loss.item(), cl_loss.item(), c_index
        ), end='\r')

    print('\ntotal_loss={:.5f}, total_ci={:.5f}'.format(np.mean(total_loss), np.mean(total_ci)))

    model.eval()
    b = 0
    total_loss_eval = []
    total_ci_eval = []
    total_pred = torch.Tensor()
    total_label = torch.Tensor()

    with torch.no_grad():
        for (prot_esm, smiles_tokens, affinity,
             drug_nodes, drug_edges, drug_batch,
             prot_nodes, prot_edge_index, prot_batch,
             drug_kg, prot_kg) in dataloader_test:

            if use_cuda:
                prot_esm = prot_esm.cuda()
                smiles_tokens = smiles_tokens.cuda()
                affinity = affinity.cuda()
                drug_nodes = drug_nodes.cuda()
                drug_edges = drug_edges.cuda()
                drug_batch = drug_batch.cuda()
                prot_nodes = prot_nodes.cuda()
                prot_edge_index = prot_edge_index.cuda()
                prot_batch = prot_batch.cuda()
                drug_kg = drug_kg.cuda()
                prot_kg = prot_kg.cuda()

            y_pred, cl_loss = model(
                smiles_tokens, drug_nodes, drug_edges, drug_batch,
                prot_esm, prot_nodes, prot_edge_index, prot_batch,
                drug_kg, prot_kg
            )

            loss = criterion(y_pred, affinity)

            out_cpu = y_pred.cpu()
            affinity_cpu = affinity.cpu()
            loss_cpu = loss.cpu().detach().item()
            c_index = ci(affinity_cpu.detach().numpy(), out_cpu.detach().numpy())

            b = b + batch_size
            total_loss_eval.append(loss_cpu)
            total_ci_eval.append(c_index)
            total_pred = torch.cat((total_pred, out_cpu), 0)
            total_label = torch.cat((total_label, affinity_cpu), 0)

            print('# [{}/{}] testing {:.1%} loss={:.5f}, ci={:.5f}'.format(
                epoch + 1, epochs, b / test_epoch_size, loss_cpu, c_index
            ), end='\r')


    y_true = total_label.detach().numpy().flatten()
    y_pred_np = total_pred.detach().numpy().flatten()

    all_ci = ci(y_true, y_pred_np)
    all_mse = mse(y_true, y_pred_np)
    all_rmse = rmse(y_true, y_pred_np)
    all_pearson = pearson(y_true, y_pred_np)
    all_spearman = spearman(y_true, y_pred_np)
    all_rm2 = rm_squared(y_true, y_pred_np)


    print('\n[Epoch {}] Eval | loss={:.5f}, CI={:.5f}, MSE={:.5f}, RMSE={:.5f}, '
          'Pearson={:.5f}, Spearman={:.5f}, r_m²={:.5f}'.format(
        epoch + 1, np.mean(total_loss_eval), all_ci, all_mse, all_rmse,
        all_pearson, all_spearman, all_rm2
    ))

    save_path = f'saved_models/CKFDTA_best_{dataset}.pkl'
    if all_ci > best_ci:
        best_ci = all_ci
        model.cpu()
        save_dict = {
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'ci': best_ci,
            'mse': all_mse,
            'epoch': epoch + 1
        }
        os.makedirs('saved_models', exist_ok=True)
        torch.save(save_dict, save_path)
        print(f'  => Best model saved (CI={best_ci:.5f})')
        if use_cuda:
            model.cuda()
