import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GCN, self).__init__()

        # Linear layer - input features
        self.linear_input = nn.Linear(input_size, hidden_sizes[0])

        # GCN layers with decreasing hidden sizes
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.conv_layers.append(GCNConv(hidden_sizes[i], hidden_sizes[i + 1]))

        # Output layers for molecular function ontology
        self.output_layers = nn.Linear(hidden_sizes[-1], output_size)

        # Dropout layers
        self.dropout_input = nn.Dropout(0.5)
        #self.dropout_hidden = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        # Linear layer - input features
        x = torch.relu(self.linear_input(x))
        x = self.dropout_input(x)

        # GCN layers with decreasing hidden sizes
        for conv_layer in self.conv_layers:
            x = torch.relu(conv_layer(x, edge_index))
            #x = self.dropout_hidden(x)

        # Aggregation step - since this is a graph level classification, the values in each layer must be made coarse to represent one graph
        x = global_mean_pool(x, batch)

        outputs = self.output_layers(x) #logits output 
        return outputs

# Initialize the model, criterion, and optimizer
input_size = 26
hidden_sizes = [812, 812, 500]  # [26, 10, 6] - gave 0.7 close accuracy
output_size = 455
model = GCN(input_size, hidden_sizes, output_size)
print(model)
torch.save(model.state_dict(), 'model_this_is_my_modell.pth')