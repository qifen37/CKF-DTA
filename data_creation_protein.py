import esm
import numpy as np
from collections import OrderedDict
import json
import torch


esm1b, esm1b_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# esm1b, esm1b_alphabet = esm.pretrained.esm1_t6_43M_UR50S()
esm1b = esm1b.eval().cpu()
esm1b_batch_converter = esm1b_alphabet.get_batch_converter()

def convert_to_tuples(seq_string: str):
    return [('', seq_string)]

def protein_embedding(prot_list, dataset):
    esm1b_contacts, token_representations = {}, {}
    i = 0
    for prot in prot_list:
        print("running at: " + str(i))
        prot = convert_to_tuples(prot)
        _, _, esm1b_batch_tokens = esm1b_batch_converter(prot)
        b = esm1b_batch_tokens.shape[1:3][0]

        if b > 1024:
            tokens = esm1b_batch_tokens[:, 0:1024]
        else:
            tokens = torch.nn.functional.pad(esm1b_batch_tokens, pad=(0, 1024 - b, 0, 0), mode='constant', value=1)

        tokens = torch.tensor(tokens.numpy())
        with torch.no_grad():
            tokens=tokens.cpu()
            results = esm1b(tokens, repr_layers=[33], return_contacts=True)
            token_representations[i] = results["representations"][33].cpu()
            contacts = esm1b.predict_contacts(tokens).cpu()
        d=torch.tensor(np.zeros([contacts.shape[0],1022])).unsqueeze(2)
        e=torch.cat([d,contacts,d],dim=2)
        d=torch.tensor(np.zeros([contacts.shape[0],1024])).unsqueeze(1)
        esm1b_contacts[i] = torch.cat([d,e,d],dim=1)
        print(i,token_representations[i].shape,esm1b_contacts[i].shape)
        np.savez('data/node/' + dataset +'.npz', dict=token_representations)
        np.savez('data/edge/' + dataset +'.npz', dict=esm1b_contacts)
        i += 1
    # torch.cuda.empty_cache()


def generate_protein_pretraining_representation(dataset, prots):
    data_dict = {}
    prots_tuple = [(str(i), prots[i][:1022]) for i in range(len(prots))]
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    i = 0
    batch = 1

    while (batch*i) < len(prots):
        print('converting protein batch: '+ str(i))
        if (i + batch) < len(prots):
            pt = prots_tuple[batch*i:batch*(i+1)]
        else:
            pt = prots_tuple[batch*i:]

        batch_labels, batch_strs, batch_tokens = batch_converter(pt)
        #model = model.cuda()
        #batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33].numpy()
        data_dict[i] = token_representations
        i += 1
    np.savez('data\DTC.npz', dict=data_dict)


dataset = 'kiba'
proteins = json.load(open('data/proteins_'+ dataset +'.txt'), object_pairs_hook=OrderedDict)
prots = []
for t in proteins.keys():
    prots.append(proteins[t])
protein_embedding(prots, dataset)
# generate_protein_pretraining_representation(dataset, prots)



