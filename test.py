import torch
import numpy as np
import os
import csv
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from torch_geometric.data import Data
from model import GCN
from testing_zone_for_pred.utils import load_predicted_PDB, seq2onehot
import scipy.sparse as sp
from preprocessing.pydataset3 import PDB_Dataset
from preprocessing.biotoolbox.structure_file_reader import build_structure_container_for_pdb
from preprocessing.biotoolbox.contact_map_builder import DistanceMapBuilder
import os
from Bio.PDB.MMCIFParser import MMCIFParser
from preprocessing.biotoolbox.contact_map_builder import DistanceMapBuilder
from preprocessing.biotoolbox.structure_file_reader import build_structure_container_for_pdb
import numpy as np

def extract_chains_from_cif(cif_file):
    parser = MMCIFParser()  # Create a CIF parser object
    structure = parser.get_structure("structure", cif_file)  # Parse the CIF file

    chains = [chain.id for model in structure for chain in model]  # Extract chain IDs
    return chains

def make_distance_maps(pdbfile, chains=None, sequence=None):
    chains = chains or extract_chains_from_cif(pdbfile)  # Extract chains from CIF file
    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)
    ca_chains = {}

    with open(pdbfile, 'r') as pdb_handle:  # Open the file in a context manager
        for chain in chains:
            pdb_handle.seek(0)  # Reset the read position to the beginning of the file
            structure_container = build_structure_container_for_pdb(pdb_handle.read(), chain).with_seqres(sequence)
            ca = mapper.generate_map_for_pdb(structure_container)
            ca_chains[chain] = ca

    return ca_chains

def write_annot_npz(pdbfile, out_dir, sequence=None):
    pdb = os.path.splitext(os.path.basename(pdbfile))[0]  # Get PDB ID without extension
    chains = extract_chains_from_cif(pdbfile)
    A_ca = make_distance_maps(pdbfile, chains, sequence=None)
    for chain in chains:
        np.savez_compressed(os.path.join(out_dir, f'{pdb}-{chain}'),
                            method=A_ca[chain].chains[chain]['method'],
                            seq=A_ca[chain].chains[chain]['seq'],
                            contact_map=A_ca[chain].chains[chain]['contact-map'])

def process_cif_directory(cif_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    for filename in os.listdir(cif_dir):
        if filename.endswith(".cif"):
            pdb_file = os.path.join(cif_dir, filename)
            try:
                write_annot_npz(pdb_file, out_dir)
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")



root = 'preprocessing/data/annot_pdb_chains_npz'
annot_file = 'preprocessing/data/nrPDB-GO_annot.tsv'
num_shards = 20

class Predictor(object):
    def __init__(self, model_file, weights_file):
        self.model_file = model_file
        self.weights_file = weights_file
        self.dataset = self.load_dataset(root, annot_file)
        self.ontology = "biological_process"
        self.load_model()

    def load_dataset(self, npz_dir, annot_file, num_shards=20):
        torch.manual_seed(12345)
        root = npz_dir
        annot_file = annot_file
        pdb_dataset = PDB_Dataset(root, annot_file, num_shards=num_shards, selected_ontology="biological_process")
        return pdb_dataset

    def load_model(self):
        input_size = 26  # Size of the one-hot encoded amino acids
        hidden_sizes = [812, 500]  # Model architecture used during training
        output_size = 700 #len(self.dataset[0].y[0])  # Output size according to the training data
        self.model = GCN(input_size, hidden_sizes, output_size)
        self.model.load_state_dict(torch.load(self.weights_file))
        self.model.eval()

    def generate_ATOM_coordinate_files(self, directory):
        pdb_files = [f for f in os.listdir(directory) if f.endswith(('.pdb', '.cif'))]
        temp_dir = os.path.join(directory, "temp_ATOM_files")
        os.makedirs(temp_dir, exist_ok=True)
        for pdb_file in pdb_files:
            pdb_id, _ = os.path.splitext(pdb_file)
            #print(pdb_id)
            structure = None
            if pdb_file.endswith('.pdb'):
                structure = PDBParser().get_structure(pdb_id, os.path.join(directory, pdb_file))
            elif pdb_file.endswith('.cif'):
                try:
                    structure = MMCIFParser().get_structure(pdb_id, os.path.join(directory, pdb_file))
                except Exception as e:
                    print(f"Error parsing CIF file {pdb_file}: {e}")
                    continue

            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()

                    if len(chain_id) > 1:
                        print(f"Skipping {pdb_id} due to chain ID '{chain_id}' exceeding PDB format limit.")
                        continue

                    output_filename = f"{pdb_id.upper()}_{chain_id}.pdb"
                    output_path = os.path.join(temp_dir, output_filename)
                    pdb_io = PDBIO()
                    pdb_io.set_structure(chain)
                    try:
                        pdb_io.save(output_path)
                        print(f"Chain {chain_id} of {pdb_id} saved to {output_path}")
                    except Exception as e:
                        print(f"Error saving PDB file {output_filename}: {e}")

        return temp_dir

    def make_cmaps(self, file):
        try:
            distances, seq = load_predicted_PDB(file)
            print(distances.shape)
        except Exception as e:
            print(f"Sequence not found for {file}. Skipping protein.")
            return None, None

        if distances is None:
            print(f"Distances not found for {file}. Skipping protein.")
            return None, None

        edge_index = self._get_adjacency_info(distances)
        if edge_index is None:
            print(f"Edge index not found for {file}. Skipping protein.")
            return None, None

        print("Edge index shape:", edge_index.shape)
        print("Edge index content:", edge_index)

        seq_onehot = seq2onehot(str(seq))
        return edge_index, seq_onehot

    def _get_adjacency_info(self, distance_matrix, threshold=10.0):
        adjacency_matrix = np.where(distance_matrix <= threshold, 1, 0)
        np.fill_diagonal(adjacency_matrix, 0)  # Ensure no self-loops
        edge_indices = np.nonzero(adjacency_matrix)

        if edge_indices[0].size == 0:
            return None

        coo_matrix = sp.coo_matrix((np.ones_like(edge_indices[0]), (edge_indices[0], edge_indices[1])))
        return torch.tensor([coo_matrix.row, coo_matrix.col], dtype=torch.long)
    
    def _load_data(self, npz_file, pdb_chain):
        pdb_file = os.path.join(npz_file)
        prot_id = npz_file.split("/")[-1]

        if os.path.isfile(pdb_file):
            cmap = np.load(pdb_file)
            sequence = str(cmap['seq'])
            ca_dist_matrix = cmap['contact_map']

            node_features = torch.tensor(seq2onehot(sequence), dtype=torch.float)
            adjacency_info = self._get_adjacency_info(ca_dist_matrix)
            length = torch.tensor(len(sequence), dtype=torch.long)
            #print(node_features, adjacency_info)
            data = Data(
                x=node_features,
                edge_index=adjacency_info,
                u=prot_id,
                length=length
            )

            return data
        else:
            print(f"File not found: {pdb_file}")
            return None
        
    def make_prediction_from_cmapdir(self, npzdir):
        predictions = []

        for root, _, files in os.walk(npzdir):
            for file in files:
                if file.endswith(".npz"):
                    npz_file = os.path.join(root, file)
                    pdb_id = os.path.splitext(file)[0]
                    try:
                        data = self._load_data(npz_file, pdb_id)
                        if data is not None:
                            with torch.no_grad():
                                output = self.model(data.x, data.edge_index, None)
                                prediction = output.sigmoid().cpu().numpy()
                                predictions.append((pdb_id, prediction))
                                #print(predictions)
                    except Exception as e:
                        print(f"Error processing {npz_file}: {e}")
        return predictions

    def make_predictions_from_pdbdir(self, adj_matrix, seqres_onehot, output_size=None):
        seqres_onehot = torch.tensor(seqres_onehot, dtype=torch.float32).unsqueeze(0)
        data = Data(x=seqres_onehot, edge_index=adj_matrix)

        with torch.no_grad():
            output = self.model(data.x, data.edge_index, None)
            predictions = output.sigmoid().cpu().numpy()

        return predictions


    def save_predictions_as_csv_cmap(self, predictions, output_file, go_numbers, go_names):
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Protein ID', 'GO Name', 'GO Term', 'Prediction Score'])
            for pdb_id, prediction_values in predictions:
                for j in range(len(prediction_values[0])):
                    writer.writerow([pdb_id, go_numbers[j], go_names[j], prediction_values[0][j]])

    def save_predictions_as_csv(self, predictions, pdb_ids, output_file, go_numbers, go_names):
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Protein ID', 'GO Name', 'GO Term', 'Prediction Score'])
            try:
                for i, pdb_id in enumerate(pdb_ids):
                    for j in range(len(predictions[i][0])):
                        writer.writerow([pdb_id, go_numbers[j], go_names[j], int(predictions[i][0][j])])
            except IndexError as e:
                print(f"Error while saving predictions: {e}")
    
    def predict(self, npz_file, output_dir):
        cmap_preds = self.make_prediction_from_cmapdir(npz_file)
        pdb_dataset = self.dataset
        go_numbers = pdb_dataset.goterms
        go_names = pdb_dataset.gonames

        self.save_predictions_as_csv_cmap(cmap_preds, output_dir,go_numbers, go_names)

        #print(cmap_preds)
        #print(output_dir)
        pass

#HPC
model_file = "hpc/hpc_model.pth"
weights_file = "hpc/GCN_weights_originalmodel"
predictor = Predictor(model_file, weights_file)
input_dir = 'alphafold_test/rice/npz'
output_file = 'output_thesis_Oryza.csv'
predictor.predict(input_dir, output_file)
