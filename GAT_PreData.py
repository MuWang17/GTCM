import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
import pandas as pd
import networkx as nx

#Loading data
datafile = "CPDB"
if datafile == "CPDB":
    string_edge = pd.read_csv('./data/CPDB/CPDB_symbols_edgelist.tsv', encoding='utf-8', sep='\t', header=0)
    savepath = "./data/CPDB/CPDB_PPI_data.tsv"
elif datafile == "STRING":
    string_edge = pd.read_csv('./data/STRING/string_edge_file.txt', encoding='utf-8', sep=' ', header=None)
    string_edge.columns = ['partner1', 'partner2', 'confidence']
    savepath = "./data/STRING/STRING_PPI_data.tsv"
else:
    string_edge = pd.read_csv('./data/PathNet/PathNet.txt', encoding='utf-8', sep=' ', header=None)
    string_edge.columns = ['partner1', 'partner2', 'confidence']
    savepath = "./data/PathNet/PathNet_PPI_data.tsv"

G = nx.from_pandas_edgelist(df=string_edge, source='partner1', target='partner2', edge_attr='confidence')
adj_pd = nx.to_pandas_adjacency(G, weight='confidence')

#Confidence score as input feature
adjacency_matrix = adj_pd.values
node_names = adj_pd.index.values
net_features = adj_pd
adj_pd_label = nx.to_pandas_adjacency(G)
label = torch.FloatTensor(adj_pd_label.values)
matrix1 = coo_matrix(adj_pd_label)
edge_index, _ = from_scipy_sparse_matrix(matrix1)
node_features = adjacency_matrix

# Creating a PyTorch Geometric Data Object
data = Data(x=torch.FloatTensor(node_features), edge_index=edge_index)

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GraphAutoencoder, self).__init__()
        # Encoder: Use two layers of GAT
        self.encoder = GATConv(num_features, hidden_dim, heads=3, dropout=0.25)
        self.encoder2 = GATConv(3*hidden_dim, hidden_dim, heads=1, concat=True, dropout=0.25)
    def encode(self, x, edge_index):
        x = F.dropout(x, p=0.5)
        x = F.relu(self.encoder(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.encoder2(x, edge_index)
        return x
    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return z

# Instantiate Model
in_channels = node_features.shape[1]  # Dimensions of input features
out_channels = 16  # The dimension of the hidden layer features
model = GraphAutoencoder(in_channels, out_channels)

# Choosing a loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
for epoch in range(200):
    optimizer.zero_grad()
    z = model(data)  # Encoding and decoding node features
    loss = criterion(z, label)         # Calculate the loss value
    loss.backward()                     # Back Propagation
    optimizer.step()                    # Update weights

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item()}')

model.eval()
with torch.no_grad():
    PPI_data = model.encode(data.x, data.edge_index)
PPI_data = torch.sigmoid(PPI_data)
PPI_data = PPI_data.detach().numpy()
PPI_data = pd.DataFrame(PPI_data, index=node_names)
PPI_data.to_csv(savepath, "\t")
