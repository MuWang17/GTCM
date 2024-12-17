import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from GTCM_model import GTCM
import pickle
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

#Parameter settings
hidden_channels = 100
dataset_file = "CPDB"
epochs = 500
lr = 0.002
w_decay = 0.00001
run_times = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Loading data
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if dataset_file == "STRING":
    ppiAdj = load_obj("./data/STRING/STRING_YC_ppi.pkl")
    goAdj = load_obj("./data/STRING/STRING_YC_go.pkl")
    pathAdj = load_obj("./data/STRING/STRING_YC_path.pkl")
    omicsfeature = load_obj("./data/STRING/STRING_16201_dataset_ten_5CV_YC.pkl")
    PrePath = "./data/STRING/predicted_socres_GTCM_STRING.txt"
elif dataset_file == "CPDB":
    ppiAdj = load_obj("./data/CPDB/CPDB_YC_ppi.pkl")
    goAdj = load_obj("./data/CPDB/CPDB_YC_go.pkl")
    pathAdj = load_obj("./data/CPDB/CPDB_YC_path.pkl")
    omicsfeature = load_obj("./data/CPDB/CPDB_13997_dataset_ten_5CV_YC.pkl")
    PrePath = "./data/CPDB/predicted_socres_GTCM_CPDB.txt"
else:
    ppiAdj = load_obj("./data/PathNet/PathNet_YC_ppi.pkl")
    goAdj = load_obj("./data/PathNet/PathNet_YC_go.pkl")
    pathAdj = load_obj("./data/PathNet/PathNet_YC_path.pkl")
    omicsfeature = load_obj("./data/PathNet/GTCM_PathNet_dataset_ten_5CV_YC.pkl")
    PrePath = "./data/PathNet/predicted_socres_GTCM_PathNet.txt"
graphlist = []
for i, network in enumerate([ppiAdj, pathAdj, goAdj]):
    std = StandardScaler()
    features = std.fit_transform(np.abs(omicsfeature['feature'].detach().numpy()))
    features = torch.FloatTensor(features)

    data = Data(x=features, y=omicsfeature['label'], edge_index=network["edge_index"], mask=omicsfeature['mask'],
                node_names=omicsfeature['node_name'])
    graphlist.append(data)
graphdata = [graph.to(device) for graph in graphlist]
data = graphdata[0]

in_channels = graphdata[0].x.shape[1]
data = data.to(device)
pred_all = np.zeros((data.num_nodes, 1))

for i in np.arange(0, run_times):
    # Creating model
    model = GTCM(in_channels, hidden_channels).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.linear1.parameters(), weight_decay=w_decay),
        dict(params=model.linear_r0.parameters(), weight_decay=w_decay),
        dict(params=model.linear_r1.parameters(), weight_decay=w_decay),
        dict(params=model.linear_r2.parameters(), weight_decay=w_decay),
        dict(params=model.linear_r3.parameters(), weight_decay=w_decay),
        dict(params=model.weight_r0, lr=lr * 0.1),
        dict(params=model.weight_r1, lr=lr * 0.1),
        dict(params=model.weight_r2, lr=lr * 0.1),
        dict(params=model.weight_r3, lr=lr * 0.1)
    ], lr=lr)

    # Training model
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        r0, r1, r2, r3, w1, w2, r4 = model(graphdata)

        l0 = F.binary_cross_entropy_with_logits(r0[data.mask], data.y[data.mask].view(-1, 1))
        l1 = F.binary_cross_entropy_with_logits(r1[data.mask], data.y[data.mask].view(-1, 1))
        l2 = F.binary_cross_entropy_with_logits(r2[data.mask], data.y[data.mask].view(-1, 1))
        l3 = F.binary_cross_entropy_with_logits(r3[data.mask], data.y[data.mask].view(-1, 1))
        l4 = F.binary_cross_entropy_with_logits(r4[data.mask], data.y[data.mask].view(-1, 1))
        loss = w1 * l0 + w2 * l1 + w2 * l2 + w2 * l3 + w1 * l4

        loss.backward()
        optimizer.step()

    if epoch % 500 == 0:
        print('Training GTCM for %d times.' % (i))

    r0, r1, r2, r3, w1, w2, r4 = model(graphdata)
    pred = torch.sigmoid(r4).cpu().detach().numpy()
    pred_all = pred + pred_all

pred_all = pred_all / run_times
pre_res = pd.DataFrame(pred_all, columns=['score'], index=data.node_names)
pre_res.sort_values(by=['score'], inplace=True, ascending=False)
# Save the final ranking list of predicted driver genes
pre_res.to_csv(path_or_buf=PrePath, sep='\t', index=True, header=True)