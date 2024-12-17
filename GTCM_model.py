import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import dropout_adj
import torch_geometric.nn as pyg_nn
import numpy as np
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScaledDotProductAttention(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        u = torch.matmul(q, k.transpose(-2, -1))
        u = u / self.scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)

        attn = self.softmax(u)
        output = torch.matmul(attn, v)

        return attn, output


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):
        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        n_q, d_q_ = q.size()
        n_k, d_k_ = k.size()
        n_v, d_v_ = v.size()

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(n_q, n_head, d_q).permute(
            1, 0, 2).contiguous().view(-1, n_q, d_q)
        k = k.view(n_k, n_head, d_k).permute(
            1, 0, 2).contiguous().view(-1, n_k, d_k)
        v = v.view(n_v, n_head, d_v).permute(
            1, 0, 2).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, n_q, d_v).permute(
            0, 1, 2).contiguous().view(n_q, -1)
        output = self.fc_o(output)

        return attn, output

class CrossModalMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_k, d_v, d_model, d_o):
        super().__init__()
        self.d_k = d_k
        self.n_heads = n_heads
        self.wq = nn.Linear(d_model, d_k)
        self.wk = nn.Linear(d_model, d_k)
        self.wv = nn.Linear(d_model, d_v)

        self.attention = MultiHeadAttention(
            n_head=n_heads, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)
        self.layer_norm = nn.LayerNorm(200)
        self.fc = nn.Linear(200, 100)

    def forward(self, modal_features, mask=None):
        attended_features = []
        q = self.wq(modal_features[0])

        for j in range(len(modal_features)):
            k = self.wk(modal_features[j])
            v = self.wv(modal_features[j])

            if mask is not None:
                attn_mask = mask[j]
            else:
                attn_mask = None

            # Apply the attention mechanism directly
            _, out = self.attention(q, k, v, mask=attn_mask)

            out = self.fc(self.layer_norm(torch.cat((modal_features[j], out), 1)))

            attended_features.append(out)

        return attended_features


class GTCM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.25,
                 weights=[0.90, 0.10]):
        super().__init__()

        self.linear = Linear(in_channels, hidden_channels)
        #GraphSAGE network layers
        self.conv_k1_1 = pyg_nn.SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels, aggr='max')
        self.conv_k2_1 = pyg_nn.SAGEConv(in_channels=2 * hidden_channels, out_channels=hidden_channels, aggr='max')
        self.conv_k3_1 = pyg_nn.SAGEConv(in_channels=3 * hidden_channels, out_channels=hidden_channels, aggr='max')

        #Transformer with cross-attention layer
        self.Mutl_attn = CrossModalMultiHeadAttention(2, d_k=32, d_v=64, d_model=hidden_channels, d_o=hidden_channels)
        self.MLP = nn.Sequential(
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.LayerNorm(100),
            nn.Linear(100, 1),
        )

        self.Relu = nn.GELU
        self.norm = nn.LayerNorm(100)

        self.lin1 = Linear(hidden_channels, 1)
        self.linear_r0 = Linear(300, 100)
        self.linear_r1 = Linear(6 * hidden_channels, 100)
        self.linear_r2 = Linear(9 * hidden_channels, 100)
        self.linear_r3 = Linear(12 * hidden_channels, 100)

        # Attention weights on outputs of different convolutional layers
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([weights[1]]), requires_grad=True)

    def forward(self, data):
        P_input = data[0].x
        edge_index_P = data[0].edge_index
        G_input = data[1].x
        edge_index_G = data[1].edge_index
        Y_input = data[2].x
        edge_index_Y = data[2].edge_index

        edge_index_1, _ = dropout_adj(edge_index_P, p=0.25,
                                      force_undirected=True,
                                      num_nodes=P_input.shape[0],
                                      training=self.training)
        edge_index_2, _ = dropout_adj(edge_index_G, p=0.25,
                                      force_undirected=True,
                                      num_nodes=G_input.shape[0],
                                      training=self.training)
        edge_index_3, _ = dropout_adj(edge_index_Y, p=0.25,
                                      force_undirected=True,
                                      num_nodes=Y_input.shape[0],
                                      training=self.training)

        P_input = F.dropout(P_input, p=0.25, training=self.training)
        G_input = F.dropout(G_input, p=0.25, training=self.training)
        Y_input = F.dropout(Y_input, p=0.25, training=self.training)


        P0 = torch.relu(self.linear(P_input))
        G0 = torch.relu(self.linear(G_input))
        Y0 = torch.relu(self.linear(Y_input))

        # layer1
        #Learning features from PPI network
        P_k1_1 = self.conv_k1_1(P0, edge_index_1)
        P1 = torch.cat((P0, torch.relu(P_k1_1)), 1)
        P1 = F.dropout(P1, p=0.25, training=self.training)

        #Learning features from pathway network
        G_k1_1 = self.conv_k1_1(G0, edge_index_2)
        G1 = torch.cat((G0, torch.relu(G_k1_1)), 1)
        G1 = F.dropout(G1, p=0.25, training=self.training)

        # Learning features from gene functional similarity network
        Y_k1_1 = self.conv_k1_1(Y0, edge_index_3)
        Y1 = torch.cat((Y0, torch.relu(Y_k1_1)), 1)
        Y1 = F.dropout(Y1, p=0.25, training=self.training)

        # layer2
        P_k2_1 = self.conv_k2_1(P1, edge_index_1)
        P2 = torch.cat((P1, torch.relu(P_k2_1)), 1)
        P2 = F.dropout(P2, p=0.25, training=self.training)

        G_k2_1 = self.conv_k2_1(G1, edge_index_2)
        G2 = torch.cat((G1, torch.relu(G_k2_1)), 1)
        # G2 = torch.relu(G_k2_1)
        G2 = F.dropout(G2, p=0.25, training=self.training)

        Y_k2_1 = self.conv_k2_1(Y1, edge_index_3)
        Y2 = torch.cat((Y1, torch.relu(Y_k2_1)), 1)
        Y2 = F.dropout(Y2, p=0.25, training=self.training)

        # layer3
        P_k3_1 = self.conv_k3_1(P2, edge_index_1)
        P3 = torch.cat((P2, torch.relu(P_k3_1)), 1)
        P3 = F.dropout(P3, p=0.25, training=self.training)

        G_k3_1 = self.conv_k3_1(G2, edge_index_2)
        G3 = torch.cat((G2, torch.relu(G_k3_1)), 1)
        G3 = F.dropout(G3, p=0.25, training=self.training)

        Y_k3_1 = self.conv_k3_1(Y2, edge_index_3)
        Y3 = torch.cat((Y2, torch.relu(Y_k3_1)), 1)
        Y3 = F.dropout(Y3, p=0.25, training=self.training)

        #Feature fusion in each layer
        R0 = torch.cat((P0, G0, Y0), 1)
        res0 = self.linear_r0(R0)
        R1 = torch.cat((P1, G1, Y1), 1)
        res1 = self.linear_r1(R1)
        R2 = torch.cat((P2, G2, Y2), 1)
        res2 = self.linear_r2(R2)
        R3 = torch.cat((P3, G3, Y3), 1)
        res3 = self.linear_r3(R3)

        #Enhance features using cross-attention
        embs = [res0, res1, res2, res3]
        embs2 = self.Mutl_attn(embs)

        emb_f = torch.cat(embs2, 1)
        # MLP classification layer
        r4 = self.MLP(emb_f)

        r0 = self.lin1(embs2[0])
        r1 = self.lin1(embs2[1])
        r2 = self.lin1(embs2[2])
        r3 = self.lin1(embs2[3])

        w1 = self.weight_r0
        w2 = self.weight_r1

        return r0, r1, r2, r3, w1, w2, r4

