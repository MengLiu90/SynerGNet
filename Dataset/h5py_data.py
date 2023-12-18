import numpy as np
import h5py
import networkx as nx
import pandas as pd
import scipy.sparse as sps
import os

df_instances = pd.read_csv('instances_list.csv')
instances_list = df_instances.instances.tolist()
df_classify = df_instances

for i in range(len(df_instances)):
    ins = df_instances.loc[i,'instances']
    print('current graph/instance being converted', ins)
    string = ins.replace('\r', '')
    cell_drug_list = string.split('_')
    cell = cell_drug_list[0]
    dr1 = cell_drug_list[1]
    dr2 = cell_drug_list[2]

    df_nodeTable = pd.read_csv(f'NodeTables/NodeTable_{cell}_{dr1}_{dr2}.csv')

    df_edgeTable = pd.read_csv(f'EdgeTables/EdgeTable_{cell}_{dr1}_{dr2}.csv')
    G = nx.from_pandas_edgelist(df_edgeTable, 'Prot1', 'Prot2', 'Score')
    G_node_ids = list(G.nodes)
    df_nodeTable = df_nodeTable.set_index('Nodes')
    nodeTable = df_nodeTable.reindex(G_node_ids)

    feature_matrix = nodeTable.to_numpy()
    A = nx.to_numpy_array(G, nodelist=G_node_ids)  # graph adjacency matrix
    spsA = sps.coo_matrix(A)
    coo_A = np.array([spsA.row, spsA.col])
    edge_index = coo_A.astype(int)
    # print(edge_index)

    edge_tuple = np.transpose(edge_index)
    edge_weight = [G.get_edge_data(G_node_ids[u], G_node_ids[v]) for (u, v) in edge_tuple]
    score = pd.DataFrame(edge_weight, columns=['Score']).to_numpy()

    lbl = df_classify[(df_classify['Cell_Line'] == cell) & (df_classify['CID_1'] == dr1) &
                        ((df_classify['CID_2'] == dr2))]['class'].values[0]
    label = int(lbl)
    # print(label)
    with h5py.File(os.path.join('Input_data/', '{}_{}_{}.h5'.format(cell, dr1, dr2)), 'w') as f:
        f.create_dataset('X', data=feature_matrix, compression="gzip", compression_opts=9)
        f.create_dataset('A', data=A, compression="gzip", compression_opts=9)
        f.create_dataset('eI', data=edge_index, compression="gzip", compression_opts=9)
        f.create_dataset('edge_weight', data=score, compression="gzip", compression_opts=9)
        f.create_dataset('y', data=np.array(label,dtype=int).reshape((1,1)), compression="gzip", compression_opts=9)


