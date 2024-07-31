from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
import csv
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T

def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x

class PDB_Dataset(Dataset):
    def __init__(self, root, annot_file, num_shards=20, selected_ontology = None, transform=None, pre_transform=None):
        annot_data = self.annot_file_reader(annot_file)
        self.prot2annot = annot_data[0]
        self.gonames = self.annot_file_reader(annot_file)[2][selected_ontology]
        self.goterms = self.annot_file_reader(annot_file)[1][selected_ontology]
        self.prot_list = [prot_id for prot_id in annot_data[3] if os.path.exists(os.path.join(root, f'{prot_id}.npz'))] #prot list only includes the ones that has the protein id 
        self.npz_dir = root
        self.num_shards = num_shards
        self.selected_ontology = selected_ontology
        self.y_labels = self.annot_file_reader(annot_file)[1][selected_ontology]
        super(PDB_Dataset, self).__init__(root, transform, pre_transform)
            
    def annot_file_reader(self, annot_filename):
        onts = ['molecular_function', 'biological_process', 'cellular_component']
        prot2annot = {}
        goterms = {ont: [] for ont in onts}
        gonames = {ont: [] for ont in onts}
        prot_list = []
        with open(annot_filename, mode='r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')

            # molecular function
            next(reader, None)  # skip the headers
            goterms[onts[0]] = next(reader)
            next(reader, None)  # skip the headers
            gonames[onts[0]] = next(reader)

            # biological process
            next(reader, None)  # skip the headers
            goterms[onts[1]] = next(reader)
            next(reader, None)  # skip the headers
            gonames[onts[1]] = next(reader)

            # cellular component
            next(reader, None)  # skip the headers
            goterms[onts[2]] = next(reader)
            next(reader, None)  # skip the headers
            gonames[onts[2]] = next(reader)

            next(reader, None)  # skip the headers
            for row in reader:
                prot, prot_goterms = row[0], row[1:]
                prot2annot[prot] = {ont: [] for ont in onts}
                for i in range(3):
                    goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if
                                    goterm != '']
                    prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]), dtype=np.int64)
                    prot2annot[prot][onts[i]][goterm_indices] = 1.0
                prot_list.append(prot)
        return prot2annot, goterms, gonames, prot_list

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.prot_list))]
    
    @property
    def num_classes(self):
        if self.selected_ontology is not None:
            return len(self.prot2annot[self.prot_list[0]][self.selected_ontology])
        else:
            return None

    def process(self):
        data_list = []
        for index, prot_id in tqdm(enumerate(self.prot_list), total=len(self.prot_list)):
            data = self._load_data(prot_id, self.prot2annot)

            if data is not None:
                data_list.append(data)
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))

        return data_list

    def _load_data(self, prot_id, prot2annot):
        pdb_file = os.path.join(self.npz_dir, f'{prot_id}.npz')

        if os.path.isfile(pdb_file):
            cmap = np.load(pdb_file)
            sequence = str(cmap['seqres'])
            ca_dist_matrix = cmap['C_alpha']
            cb_dist_matrix = cmap['C_beta']

            node_features = torch.tensor(seq2onehot(sequence), dtype=torch.float)
            adjacency_info = self._get_adjacency_info(ca_dist_matrix)
            length = torch.tensor(len(sequence), dtype=torch.long)
            labels = self._get_labels(prot_id, prot2annot)

            if self.selected_ontology is not None:
                try:
                    labels = labels[self.selected_ontology]
                except:
                    print("The ontology you provided doesn't exists in the presets")

            data = Data(
                x=node_features,
                edge_index=adjacency_info,
                u=prot_id,
                y=labels,
                length=length
            )

            return data
        else:
            print(f"File not found: {pdb_file}")
            return None

    def _get_labels(self, prot_id, prot2annot):
        mf_labels = torch.tensor(prot2annot[prot_id]['molecular_function'], dtype=torch.long).unsqueeze(0)
        bp_labels = torch.tensor(prot2annot[prot_id]['biological_process'], dtype=torch.long).unsqueeze(0)
        cc_labels = torch.tensor(prot2annot[prot_id]['cellular_component'], dtype=torch.long).unsqueeze(0)

        # Check for empty tensors and handle individually
        if mf_labels.numel() == 0:
            mf_labels = torch.zeros(1, dtype=torch.long)
        if bp_labels.numel() == 0:
            bp_labels = torch.zeros(1, dtype=torch.long)
        if cc_labels.numel() == 0:
            cc_labels = torch.zeros(1, dtype=torch.long)

        return {
            'molecular_function': mf_labels,
            'biological_process': bp_labels,
            'cellular_component': cc_labels
        }


    def _get_adjacency_info(self, distance_matrix, threshold=8.0):
        adjacency_matrix = np.where(distance_matrix <= threshold, 1, 0)
        np.fill_diagonal(adjacency_matrix, 0)  # Ensure no self-loops
        edge_indices = np.nonzero(adjacency_matrix)

        coo_matrix = sp.coo_matrix((np.ones_like(edge_indices[0]), (edge_indices[0], edge_indices[1])))
        return torch.tensor([coo_matrix.row, coo_matrix.col], dtype=torch.long)

    def len(self):
        return len(self.prot_list)
    
    def get(self, idx): # cannot return none
        prot_id = self.prot_list[idx]
        data = self._load_data(prot_id, self.prot2annot)
        return data
        
    def print_batch_details(self, batch):
        print("Batch Details:")
        print("Number of Graphs in Batch:", batch.num_graphs)
        print("Number of Nodes:", batch.num_nodes)
        print("Number of Edges:", batch.num_edges)
        print("Node Features Shape:", batch.x.shape)
        print("Edge Index Shape:", batch.edge_index.shape)
        
