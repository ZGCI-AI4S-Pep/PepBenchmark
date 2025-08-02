from ogb.utils import smiles2graph
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import torch
import json
import os

def seq_to_mol(sequence):
    """Convert peptide sequence to RDKit Mol object"""
    return Chem.MolFromSequence(sequence)

def seq_to_smiles(sequence):
    """Convert peptide sequence to SMILES representation"""
    mol = seq_to_mol(sequence)
    return Chem.MolToSmiles(mol) if mol else ""

def process_sequence(sequence):
    """Process peptide sequence to generate graph data"""
    smiles = seq_to_smiles(sequence)
    if not smiles:
        return None
    graph_data = smiles2graph(smiles)
    return graph_data

def datasets(f_path, is_smiles=False):
    if is_smiles:
        fasta = pd.read_csv(f_path+'smiles.csv', sep="\t", header=None, names=["feature"])
        all_label = pd.read_csv(f_path+'label.csv', sep="\t", header=None, names=["label"])
        smiles = []
        labels = []
        for seq in tqdm(range(len(fasta["feature"][1:]))):
            smiles.append(fasta["feature"][seq+1])
            labels.append(float(all_label["label"][seq+1]))
    else:
        fasta = pd.read_csv(f_path, sep="\t", header=None, names=["sequence"])
        # labels = pd.read_csv(f_path, sep="\t", header=None, names=["label"])
        smiles = []
        labels = []
        # print(labels)
        for seq in tqdm(fasta["sequence"][1:]):
            # print(seq.split(',')[0])
            smiles.append(seq_to_smiles(seq.split(',')[0]))
            labels.append(int(seq.split(',')[1]))

    return smiles, labels

class PeptideDataset(Dataset):
    def __init__(self, smiles, labels, partition, seed, mode, gpath):
        super(PeptideDataset, self).__init__()
        self.smiles = smiles
        # print(labels)
        self.labels = torch.tensor(labels)
        print(partition)
        with open(partition,'r') as f:
            self.partition = json.load(f)[seed][mode]
        self.gpath = gpath
        self._process()


    def _process(self):
        if os.path.exists(self.gpath+'graphs.pt'):
            print(f'Loading graphs from {self.gpath}')
            self.all_graphs = torch.load(self.gpath+'graphs.pt', weights_only=False)
        else:
            print(f'Processing graphs and saving to {self.gpath}')
            self.all_graphs = []
            for i in tqdm(range(len(self.smiles))):
                graph_data = smiles2graph(self.smiles[i])
                if graph_data and self.labels[i] is not None:
                    graph = Data(
                        x=torch.from_numpy(graph_data['node_feat']),
                        edge_index=torch.from_numpy(graph_data['edge_index']),
                        edge_attr=torch.from_numpy(graph_data['edge_feat'])
                    )
                    graph.y = self.labels[i]
                else:
                    graph = None
                self.all_graphs.append(graph)
            os.makedirs(os.path.dirname(self.gpath), exist_ok=True)
            torch.save(self.all_graphs, self.gpath+'graphs.pt')
        self.graphs = [self.all_graphs[i] for i in self.partition if self.all_graphs[i] is not None]
                

    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        return self.graphs[idx]



if __name__ == "__main__":
    # # Example usage
    # peptide_sequence = "ACDEFGHIKLMNPQRSTVWY"  # Example peptide sequence
    # graph_data = process_sequence(peptide_sequence)
    
    # if graph_data:
    #     print("Graph data generated successfully.")
    #     print("Nodes:", graph_data['node_feat'])
    #     print("Edges:", graph_data['edge_index'])
    #     print("Edge Features:", graph_data['edge_feat'])
    #     print("Number of nodes:", graph_data['num_nodes'])
    #     print(graph_data)
    # else:
    #     print("Failed to generate graph data from the peptide sequence.")
    smiles, labels = datasets("/data0/data_share/peptide_dataset/processed_2025.6.06v/Theraputic-AMP/AF_APML/Random_split/random1/train.csv")  # Replace with your actual file path
    train_dataset = PeptideDataset(smiles, labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for batch in train_loader:
        print(batch)
        # break  # Just to show the first batch