import numpy as np
import scipy.sparse as sp
import torch

from utils import encode_onehot, normalize_features, normalize_adj


class Dataset:
    def __init__(self, source, name):
        self.source = source
        self.name = name

    def load_data(self):
        if self.name == "cora":
            content_path = self.source + '/cora.content'
            cite_path = self.source + '/cora.cites'

            idx_features_labels = np.genfromtxt(content_path, dtype=np.dtype(str))
            features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
            labels = encode_onehot(idx_features_labels[:, -1])

            idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
            idx_map = {j: i for i, j in enumerate(idx)}
            edges_unordered = np.genfromtxt(cite_path, dtype=np.int32)
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32)
            '''
            for i in range(adj.shape[0]):
                for j in range(i, adj.shape[1]):
                    adj[i][j] = adj[j][i] = max(adj[i][j], adj[j][i]
            '''
            # build symmetric adjacency matrix
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

            features = normalize_features(features)
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))

            idx_train = range(140)
            idx_val = range(200, 500)
            idx_test = range(500, 1500)

            adj = torch.FloatTensor(np.array(adj.todense()))
            features = torch.FloatTensor(np.array(features.todense()))
            labels = torch.LongTensor(np.where(labels)[1])

            idx_train = torch.LongTensor(idx_train)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)

            return adj, features, labels, idx_train, idx_val, idx_test
