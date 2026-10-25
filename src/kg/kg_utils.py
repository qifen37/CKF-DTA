# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np


def load_kg_triples(kg_path, sep='\t', columns=None):
    if columns is None:
        columns = ['head', 'relation', 'tail']

    df = pd.read_csv(kg_path, sep=sep, header=None)
    df.columns = columns[:len(df.columns)]
    return df[['head', 'relation', 'tail']]


def remove_dti_from_kg(kg_df, dti_relation='drug_target_interaction'):
    dti_triples = kg_df[kg_df['relation'] == dti_relation]
    kg_clean = pd.concat([kg_df, dti_triples]).drop_duplicates(
        subset=['head', 'relation', 'tail'], keep=False
    )
    kg_clean.index = range(len(kg_clean))
    return kg_clean, dti_triples


def merge_kgs(*kg_dfs):
    merged = pd.concat(kg_dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=['head', 'relation', 'tail'])
    merged.index = range(len(merged))
    return merged


def save_triples_tsv(kg_df, output_path):
    kg_df[['head', 'relation', 'tail']].to_csv(
        output_path, sep='\t', header=False, index=False
    )
    print(f"Saved {len(kg_df)} triples to {output_path}")


def load_entity_ids(mapping_path, prefix):
    entity_ids = []
    with open(mapping_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            entity_ids.append(f"{prefix}{idx}")
    return entity_ids


def build_entity_mapping(drug_smiles_list, protein_id_list,
                         drug_mapping_path=None, protein_mapping_path=None):
    drug_to_kg_id = {}
    for idx, smiles in enumerate(drug_smiles_list):
        drug_to_kg_id[smiles] = f"Drug::{idx}"

    protein_to_kg_id = {}
    for idx, pid in enumerate(protein_id_list):
        protein_to_kg_id[pid] = f"Protein::{idx}"

    return drug_to_kg_id, protein_to_kg_id


def load_and_merge_kgs(hetionet_path=None, biokg_path=None, yamanishi_path=None,
                       output_path=None):
    kgs = []

    if hetionet_path and os.path.exists(hetionet_path):
        kg = load_kg_triples(hetionet_path, sep='\t')
        kg, _ = remove_dti_from_kg(kg)
        kgs.append(kg)
        print(f"Loaded Hetionet: {len(kg)} triples")

    if biokg_path and os.path.exists(biokg_path):
        kg = load_kg_triples(biokg_path, sep=',')
        kg, _ = remove_dti_from_kg(kg)
        kgs.append(kg)
        print(f"Loaded BioKG: {len(kg)} triples")

    if yamanishi_path and os.path.exists(yamanishi_path):
        kg = load_kg_triples(yamanishi_path, sep='\t')
        kg, _ = remove_dti_from_kg(kg)
        kgs.append(kg)
        print(f"Loaded Yamanishi08: {len(kg)} triples")

    if not kgs:
        raise ValueError("No KG data files found. Provide at least one KG path.")

    merged_kg = merge_kgs(*kgs)
    print(f"Merged KG: {len(merged_kg)} triples (deduplicated)")

    if output_path:
        save_triples_tsv(merged_kg, output_path)

    return merged_kg
