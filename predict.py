import torch
from preprocessing.pydataset3 import PDB_Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model import GCN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Define model architecture and load pre-trained weights
input_size = len(dataset[0].x[0])
hidden_sizes = [812, 500]
output_size = len(dataset[0].y)
model = GCN(input_size, hidden_sizes, output_size)
model.load_state_dict(torch.load('new_weights.pth'))
model.to(device)
model.eval()

# Create DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Predictions
all_labels = []
all_roc_auc = []
all_pr_auc = []

with torch.no_grad():
    for data in tqdm(test_loader, desc='Testing'):
        data = data.to(device)
        outputs = model(data.x, data.edge_index, data.batch)
        predictions = torch.sigmoid(outputs).cpu().numpy()
        labels = data.y.cpu().numpy()

        # Calculate ROC AUC for each label
        for i in range(output_size):
            fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            all_labels.append(i)
            all_roc_auc.append(roc_auc)

        # Calculate PR AUC for each label
        for i in range(output_size):
            precision, recall, _ = precision_recall_curve(labels[:, i], predictions[:, i])
            pr_auc = auc(recall, precision)
            all_pr_auc.append(pr_auc)

# Create a DataFrame for AUC values
auc_df = pd.DataFrame({'Label': all_labels, 'ROC AUC': all_roc_auc, 'PR AUC': all_pr_auc})

# Plot violin plots
plt.figure(figsize=(10, 6))
sns.violinplot(x='Label', y='ROC AUC', data=auc_df)
plt.title('Distribution of ROC AUC for each label')
plt.xlabel('Label')
plt.ylabel('ROC AUC')
plt.savefig('roc_auc_violin_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Label', y='PR AUC', data=auc_df)
plt.title('Distribution of PR AUC for each label')
plt.xlabel('Label')
plt.ylabel('PR AUC')
plt.savefig('pr_auc_violin_plot.png')
plt.show()

# Plot ROC and PR curves for each label
for i in range(output_size):
    fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
    precision, recall, _ = precision_recall_curve(labels[:, i], predictions[:, i])

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {all_roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Label {i}')
    plt.legend()
    plt.savefig(f'roc_curve_label_{i}.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {all_pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Label {i}')
    plt.legend()
    plt.savefig(f'pr_curve_label_{i}.png')
    plt.show()
