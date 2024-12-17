import numpy as np
import torch
from sklearn import metrics
import torch.nn.functional as F
import pickle
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from GTCM_model import GTCM
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Parameter settings
hidden_channels = 100
dataset_file = "CPDB"
epochs = 500
lr = 0.002
w_decay = 0.00001

#Loading data
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if dataset_file == "STRING":
    ppiAdj = load_obj("./data/STRING/STRING_YC_ppi.pkl")
    goAdj = load_obj("./data/STRING/STRING_YC_go.pkl")
    pathAdj = load_obj("./data/STRING/STRING_YC_path.pkl")
    omicsfeature = load_obj("./data/STRING/STRING_16201_dataset_ten_5CV_YC.pkl")
elif dataset_file == "CPDB":
    ppiAdj = load_obj("./data/CPDB/CPDB_YC_ppi.pkl")
    goAdj = load_obj("./data/CPDB/CPDB_YC_go.pkl")
    pathAdj = load_obj("./data/CPDB/CPDB_YC_path.pkl")
    omicsfeature = load_obj("./data/CPDB/CPDB_13997_dataset_ten_5CV_YC.pkl")
else:
    ppiAdj = load_obj("./data/PathNet/PathNet_YC_ppi.pkl")
    goAdj = load_obj("./data/PathNet/PathNet_YC_go.pkl")
    pathAdj = load_obj("./data/PathNet/PathNet_YC_path.pkl")
    omicsfeature = load_obj("./data/PathNet/GTCM_PathNet_dataset_ten_5CV_YC.pkl")
graphlist = []
for i, network in enumerate([ppiAdj, pathAdj, goAdj]):
    std = StandardScaler()
    features = std.fit_transform(np.abs(omicsfeature['feature'].detach().numpy()))
    features = torch.FloatTensor(features)

    data = Data(x=features, y=omicsfeature['label'], edge_index=network["edge_index"], mask=omicsfeature['split_set'],
                node_names=omicsfeature['node_name'])
    graphlist.append(data)
graphdata = [graph.to(device) for graph in graphlist]
data = graphdata[0]

in_channels = graphdata[0].x.shape[1]

def test(data, mask, graphdata):
    model.eval()
    r0, r1, r2, r3, w1, w2, r4 = model(graphdata)

    pred = torch.sigmoid(r4[mask])
    precision, recall, _thresholds = metrics.precision_recall_curve(data.y[mask].cpu().numpy(),
                                                                    pred.cpu().detach().numpy())
    aupr = metrics.auc(recall, precision)
    auc = metrics.roc_auc_score(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy())
    return auc, aupr, pred.cpu().detach().numpy()


def train(tr_mask, data, graphdata):
    # Training model
    model.train()
    optimizer.zero_grad()
    r0, r1, r2, r3, w1, w2, r4 = model(graphdata)

    l0 = F.binary_cross_entropy_with_logits(r0[tr_mask], data.y[tr_mask].view(-1, 1))
    l1 = F.binary_cross_entropy_with_logits(r1[tr_mask], data.y[tr_mask].view(-1, 1))
    l2 = F.binary_cross_entropy_with_logits(r2[tr_mask], data.y[tr_mask].view(-1, 1))
    l3 = F.binary_cross_entropy_with_logits(r3[tr_mask], data.y[tr_mask].view(-1, 1))
    l4 = F.binary_cross_entropy_with_logits(r4[tr_mask], data.y[tr_mask].view(-1, 1))
    loss = w1 * l0 + w2 * l1 + w2 * l2 + w2 * l3 + w1 * l4

    loss.backward()
    optimizer.step()


# Ten times of 5_CV
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))
pred_all = []
label_all = []
file_save_path = './data/picture/'


for i in range(10):
    for cv_run in range(5):
        tr_mask, te_mask = data.mask[i][cv_run]
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

        for epoch in range(1, epochs + 1):
            train(tr_mask, data, graphdata)
            if epoch % 500 == 0:
                print(f'Training epoch: {epoch:03d}')

        AUC[i][cv_run], AUPR[i][cv_run], pred = test(data, te_mask, graphdata)
        pred_all.append(pred)
        label_all.append(data.y[te_mask].cpu().numpy())
        print('Round--%d CV--%d  AUC: %.5f, AUPR: %.5f' % (i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run]))
    print('Round--%d Mean AUC: %.5f, Mean AUPR: %.5f' % (i, np.mean(AUC[i, :]), np.mean(AUPR[i, :])))

print('GTCM 10 rounds for 5CV-- Mean AUC: %.4f, Mean AUPR: %.4f' % (AUC.mean(), AUPR.mean()))
torch.save(pred_all, os.path.join(file_save_path, 'pred_all.pkl'))
torch.save(label_all, os.path.join(file_save_path, 'label_all.pkl'))
