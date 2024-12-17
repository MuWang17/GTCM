import os
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_networkx
import pickle


cuda = torch.cuda.is_available()

def get_node_genelist():
    """
    generate gene list
    """
    print('Get gene list')
    ppi = pd.read_csv("./data/graph/CPDB_symbols_edgelist.txt", sep='\t', encoding='utf8')
    ppi.columns = ['source', 'target']

    ppi = ppi[ppi['source'] != ppi['target']]

    ppi.dropna(inplace=True)
    final_gene_node = sorted(
        list(set(ppi.source) | set(ppi.target)))

    G = nx.from_pandas_edgelist(ppi)
    ppi_df = nx.to_pandas_adjacency(G)

    temp = pd.DataFrame(index=final_gene_node, columns=final_gene_node)

    ppi_adj = temp.combine_first(ppi_df)
    ppi_adj.fillna(0, inplace=True)
    ppi_final = ppi_adj[final_gene_node].loc[final_gene_node]

    return final_gene_node, ppi_final


def get_node_omicfeature():
    """
    generate omic data
    """
    final_gene_node, _ = get_node_genelist()

    # process the omic data
    omics_file = pd.read_csv(
        './data/graph/TMPmultiomics_features.tsv', sep='\t', index_col=0)

    expendgene = sorted(list(set(omics_file.index) | set(final_gene_node)))
    temp = pd.DataFrame(index=expendgene, columns=omics_file.columns)
    omics_adj = temp.combine_first(omics_file)
    omics_adj.fillna(0, inplace=True)
    omics_adj = omics_adj.loc[final_gene_node]
    omics_adj.sort_index(inplace=True)

    # chosen 16 cancer type
    chosen_project = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD',
                      'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    omics_temp = [omics_adj[omics_adj.columns[omics_adj.columns.str.contains(
        cancer)]] for cancer in chosen_project]
    omics_data = pd.concat(omics_temp, axis=1)

    omics_feature_vector = sp.csr_matrix(omics_data, dtype=np.float32)
    omics_feature_vector = torch.FloatTensor(
        np.array(omics_feature_vector.todense()))
    print(
        f'The shape of omics_feature_vector:{omics_feature_vector.shape}')
    omics_data.to_csv("./data/graph/CPDB_YC_omics_data.tsv", sep='\t')

    return omics_feature_vector, final_gene_node


def generate_graph(thr_go, thr_path):
    """
    generate graph: PPI Pathway GO_network
    """
    print('generate graph')
    final_gene_node, ppi = get_node_genelist()

    path = pd.read_csv(os.path.join(
        './data/graph/pathsim_matrix.csv'), sep='\t', index_col=0)
    path_matrix = path.applymap(lambda x: 0 if x < thr_path else 1)
    np.fill_diagonal(path_matrix.values, 0)

    go = pd.read_csv('./data/graph/GOSemSim_matrix.csv',
                     sep='\t', index_col=0)
    go_matrix = go.applymap(lambda x: 0 if x < thr_go else 1)
    np.fill_diagonal(go_matrix.values, 0)

    networklist = []
    temp = pd.DataFrame(index=final_gene_node, columns=final_gene_node).fillna(0).astype('int8')
    for matrix in [go_matrix, path_matrix]:
        network = temp.copy()
        network.update(matrix)
        network_adj = network.loc[final_gene_node, final_gene_node]

        # network_adj = network[final_gene_node].loc[final_gene_node]
        networklist.append(network_adj)
        print('The shape of network_adj:', network_adj.shape)

    # Save the processed graph data and omic data
    ppi.to_csv(os.path.join(
        "./data/graph/CPDB/" + 'CPDB_YC_ppi.tsv'), sep='\t')
    networklist[0].to_csv(os.path.join(
        "./data/graph/CPDB/" + 'CPDB_YC_go.tsv'), sep='\t')
    networklist[1].to_csv(os.path.join(
        "./data/graph/CPDB/" + 'CPDB_YC_path.tsv'), sep='\t')

    return ppi, networklist[0], networklist[1]


def load_featured_graph(network, omicfeature):

    omics_feature_vector = sp.csr_matrix(omicfeature, dtype=np.float32)
    omics_feature_vector = torch.FloatTensor(
        np.array(omics_feature_vector.todense()))
    print(
        f'The shape of omics_feature_vector:{omics_feature_vector.shape}')

    if network.shape[0] == network.shape[1]:
        G = nx.from_pandas_adjacency(network)
    else:
        G = nx.from_pandas_edgelist(network)

    G_adj = nx.convert_node_labels_to_integers(
        G, ordering='sorted', label_attribute='label')

    print(f'If the graph is connected graph: {nx.is_connected(G_adj)}')
    print(
        f'The number of connected components: {nx.number_connected_components(G_adj)}')

    graph = from_networkx(G_adj)
    assert graph.is_undirected() == True
    print(f'The edge index is {graph.edge_index}')

    graph.x = omics_feature_vector

    return graph

#Processing Data
omicsfeature, final_gene_node = get_node_omicfeature()
ppi_network, go_network, path_network = generate_graph(0.8, 0.6)

ppi_network = pd.read_csv("./data/graph/CPDB_YC_ppi.tsv", sep='\t', index_col=0)
go_network = pd.read_csv("./data/graph/CPDB_YC_go.tsv", sep='\t', index_col=0)
path_network = pd.read_csv("./data/graph/CPDB_YC_path.tsv", sep='\t', index_col=0)
omicsfeature = pd.read_csv("./data/graph/CPDB_ppiYC_omics_data.tsv", sep='\t', index_col=0)
final_gene_node = list(omicsfeature.index)

#Save data
featured_graph = load_featured_graph(ppi_network, omicsfeature)
dataset1=dict()
dataset1['feature'] = featured_graph.x
dataset1['edge_index'] = featured_graph.edge_index
print(dataset1['edge_index'])
with open('./data/graph/CPDB_YC_ppi.pkl', 'wb') as f:
    pickle.dump(dataset1, f, pickle.HIGHEST_PROTOCOL)

featured_graph = load_featured_graph(go_network, omicsfeature)
dataset2=dict()
dataset2['feature'] = featured_graph.x
dataset2['edge_index'] = featured_graph.edge_index
print(dataset2['edge_index'])
with open('./data/graph/CPDB_YC_go.pkl', 'wb') as f:
    pickle.dump(dataset2, f, pickle.HIGHEST_PROTOCOL)

featured_graph = load_featured_graph(path_network, omicsfeature)
dataset5=dict()
dataset5['feature'] = featured_graph.x
dataset5['edge_index'] = featured_graph.edge_index
print(dataset5['edge_index'])
with open('./data/graph/CPDB_YC_path.pkl', 'wb') as f:
    pickle.dump(dataset5, f, pickle.HIGHEST_PROTOCOL)






