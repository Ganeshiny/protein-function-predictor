import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing.pydataset3 import PDB_Dataset
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix


def plot_accuracies(train_accuracies, val_accuracies, test_accuracies):
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=train_accuracies, label='Training Accuracy')
    sns.lineplot(x=epochs, y=val_accuracies, label='Validation Accuracy')
    sns.lineplot(x=epochs, y=test_accuracies, label='Test Accuracy')

    plt.title('Training, Validation, and Test Accuracies Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Function to plot ROC and PR curves
def plot_roc_pr_curves(model, test_loader, device):
    model.eval()
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Plotting ROC and PR Curves'):
            data = data.to(device)

            # Move the model to the same device as the input data
            model.to(device)

            outputs = model(data.x, data.edge_index, data.batch)

            all_test_preds.extend(torch.sigmoid(outputs.view(-1)).cpu())
            all_test_labels.extend(data.y.view(-1).cpu())

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(torch.vstack(all_test_labels), torch.vstack(all_test_preds))
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    # Calculate precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(torch.vstack(all_test_labels), torch.vstack(all_test_preds))
    pr_auc = auc(recall, precision)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend()
    plt.show()


# Function to plot confusion matrix
def plot_confusion_matrix(model, test_loader, device):
    model.eval()
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Calculating Confusion Matrix'):
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)

            all_test_preds.extend(torch.sigmoid(outputs.view(-1)).cpu())
            all_test_labels.extend(data.y.view(-1).cpu())

    # Convert probabilities to binary predictions
    test_binary_preds = (torch.vstack(all_test_preds) > 0.5).int()

    # Calculate confusion matrix
    cm = confusion_matrix(torch.vstack(all_test_labels), test_binary_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()