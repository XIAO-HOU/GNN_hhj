import numpy as np
import scipy.sparse as sp

from collections import defaultdict

class Dataset:
    def __init__(self, source, name):
        self.source = source
        self.name = name

    def load_data(self):
        if self.name == 'cora':
            content_path = self.source + '/cora.content'
            cite_path = self.source + '/cora.cites'

        features = []
        labels = []
        node_map = {}
        label_map = {}

        with open(content_path) as f:
            for i, line in enumerate(f):
                info = line.strip().split()
                features.append([float(x) for x in info[1:-1]])
                node_map[info[0]] = i
                if info[-1] not in label_map:
                    label_map[info[-1]] = len(label_map)
                labels.append(label_map[info[-1]])
        features = np.asarray(features)
        labels = np.asarray(labels)

        adj_list = defaultdict(set)
        with open(cite_path) as f:
            for i, line in enumerate(f):
                info = line.strip().split()
                assert len(info) == 2
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                adj_list[paper1].add(paper2)
                adj_list[paper2].add(paper1)

        train_ids, val_ids, test_ids = self._split_data(features.shape[0])

        return features, labels, adj_list, train_ids, val_ids, test_ids

    def _split_data(self, nodes_num, test_split=3, val_split=6):
        rand_indices = np.random.permutation(nodes_num)

        test_size = nodes_num // test_split
        val_size = nodes_num // val_split

        test_ids = rand_indices[: test_size]
        val_ids = rand_indices[test_size: test_size + val_size]
        train_ids = rand_indices[test_size + val_size:]

        return train_ids, val_ids, test_ids
