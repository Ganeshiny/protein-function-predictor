import torch
from preprocessing.pydataset3 import PDB_Dataset
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


def calculate_class_weights(dataset, device):
    # Calculate the number of classes in the dataset
    num_classes = dataset[0].y.size(1)
    print("Number of classes:", num_classes)

    # Initialize class counters
    class_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)


    # Count the number of examples in each class
    for data in dataset:
        class_counts += data.y.sum(dim=0).float().to(device)
        #print(class_counts)
        

    # Calculate class weights by taking the inverse of class frequency
    class_weights = 1.0 / (class_counts / class_counts.sum())
    print(class_weights)
    return class_weights.to(device)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing.pydataset3 import PDB_Dataset
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from balance_classes import calculate_class_weights
from focal_loss import FocalLoss
from model import GCN
from plot_metrics import plot_accuracies, plot_confusion_matrix, plot_roc_pr_curves

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

threshold = 0.5

# Set up the dataset
root = 'preprocessing/data/annot_pdb_chains_npz'
annot_file = 'preprocessing/data/nrPDB-GO_annot.tsv'
num_shards = 20

# Load data using DataLoader directly
dataset = PDB_Dataset(root, annot_file, num_shards=num_shards, selected_ontology="biological_process")
torch.manual_seed(12345)

# Splitting the dataset into train, test, and validation sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=12345)
val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=54321)

print(
    f"len train_dataset = {len(train_dataset)}, len test_dataset = {len(test_dataset)}, len validation_dataset = {len(val_dataset)}"
)
# Creating DataLoader objects out of the train, test, and validation datasets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Use the DataLoader to calculate class weights
alpha_weights = calculate_class_weights(train_dataset, device)