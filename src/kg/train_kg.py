# -*- coding: utf-8 -*-

import os
import torch
import numpy as np


def train_distmult(triples_path, save_dir, embedding_dim=400, epochs=150,
                   batch_size=1024, device=None):
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training = TriplesFactory.from_path(triples_path)

    result = pipeline(
        training=training,
        model='DistMult',
        device=device,
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs, batch_size=batch_size),
    )

    os.makedirs(save_dir, exist_ok=True)
    result.save_to_directory(save_dir)
    print(f"KG model saved to {save_dir}")

    return result


def load_kg_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.load(model_path, map_location=device)
    return result


def extract_entity_embeddings(result):
    embeddings = result.entity_representations[0](indices=None)
    entity_to_id = result.training.entity_to_id
    return embeddings, entity_to_id


def extract_drug_protein_embeddings(result, drug_ids, protein_ids):
    embeddings, entity_to_id = extract_entity_embeddings(result)
    embeddings_np = embeddings.cpu().detach().numpy()

    drug_embs = []
    for drug_id in drug_ids:
        if drug_id in entity_to_id:
            idx = entity_to_id[drug_id]
            drug_embs.append(embeddings_np[idx])
        else:
            drug_embs.append(np.zeros(embeddings_np.shape[1], dtype=np.float32))
    drug_embeddings = np.array(drug_embs)

    prot_embs = []
    for prot_id in protein_ids:
        if prot_id in entity_to_id:
            idx = entity_to_id[prot_id]
            prot_embs.append(embeddings_np[idx])
        else:
            prot_embs.append(np.zeros(embeddings_np.shape[1], dtype=np.float32))
    protein_embeddings = np.array(prot_embs)

    return drug_embeddings, protein_embeddings


def train_or_load(triples_path, save_dir, embedding_dim=400, epochs=150,
                  batch_size=1024, device=None):
    model_path = os.path.join(save_dir, 'trained_model.pkl')

    if not os.path.exists(model_path):
        print(f"KG model not found at {model_path}, training...")
        result = train_distmult(triples_path, save_dir, embedding_dim,
                                epochs, batch_size, device)
    else:
        print(f"Loading existing KG model from {model_path}")
        result = load_kg_model(model_path, device)

    return result
