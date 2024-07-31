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
print(f"debug: alpha_weights:  {alpha_weights}")

# Initialize the model, criterion, and optimizer
input_size = len(dataset[0].x[0])
hidden_sizes = [812, 500]  # [26, 10, 6] - gave 0.7 close accuracy
output_size = dataset.num_classes
model = GCN(input_size, hidden_sizes, output_size)
model.to(device)
torch.save(model.state_dict(), 'model_this_is_my_modell.pth')

criterion = FocalLoss(alpha=alpha_weights)

optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

num_epochs = 1000
train_losses = []
train_accuracies = []
val_accuracies = []
test_accuracies = []

best_val_accuracy = 0.0  # Initialize the best validation accuracy

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index, data.batch)
        targets = data.y.float()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Calculate training accuracy
        all_train_preds.extend(torch.sigmoid(outputs.view(-1)).cpu())
        all_train_labels.extend(data.y.view(-1).cpu())

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracy = accuracy_score(torch.vstack(all_train_labels), (torch.vstack(all_train_preds) > threshold).int())
    train_accuracies.append(train_accuracy.item())

    # Validation
    model.eval()
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for data in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)

            all_val_preds.extend(torch.sigmoid(outputs.view(-1)).cpu())
            all_val_labels.extend(data.y.view(-1).cpu())

    val_accuracy = accuracy_score(torch.vstack(all_val_labels),
                                    (torch.vstack(all_val_preds) > threshold).int())
    val_accuracies.append(val_accuracy.item())

    # Testing
    model.eval()
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Testing'):
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)

            all_test_preds.extend(torch.sigmoid(outputs.view(-1)).cpu())
            all_test_labels.extend(data.y.view(-1).cpu())

    test_accuracy = accuracy_score(torch.vstack(all_test_labels),
                                    (torch.vstack(all_test_preds) > threshold).int())
    test_accuracies.append(test_accuracy.item())

    print(f'Epoch {epoch + 1}/{num_epochs} - '
          f'Training Loss: {avg_train_loss:.4f}, '
          f'Training Accuracy: {train_accuracy:.4f}, '
          f'Validation Accuracy: {val_accuracy:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}')

    # Check if the current model has the best validation accuracy
    if val_accuracy > best_val_accuracy:
        print(f'New best validation accuracy found: {val_accuracy:.4f}. Saving model weights...')
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'new_weights_02.pth')

# Plot accuracies
plot_accuracies(train_accuracies, val_accuracies, test_accuracies)

# Plot ROC and PR curves
plot_roc_pr_curves(model, test_loader, device)

# Plot confusion matrix
plot_confusion_matrix(model, test_loader, device)