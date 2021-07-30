import random
import torch

from torch import nn
from myGraphSAGE.layers import SageLayer


class GraphSage(nn.Module):
    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.agg_func = agg_func

        self.raw_features = raw_features
        self.adj_lists = adj_lists

        for index in range(1, num_layers + 1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

    def forward(self, nodes_batch):
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        for i in range(self.num_layers):
            lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict = self._get_unique_neighs_list(lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))
        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]  # node_batch_layers = [layer1, layer0, layer_center]
            pre_neighs = nodes_batch_layers[index-1]
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer'+str(index))

            if index > 1:
                nb = self._node_map(nb, pre_neighs)

            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb], aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if num_sample is not None:
            samp_neighs = [set(random.sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | {nodes[i]} for i, samp_neigh in enumerate(samp_neighs)]
        unique_node_list = list(set.union(*samp_neighs))
        unique_nodes_dict = {node: i for i, node in enumerate(unique_node_list)}
        return unique_node_list, samp_neighs, unique_nodes_dict

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs):
        unique_node_list, samp_neighs, unique_nodes_dict = pre_neighs
        assert len(nodes) == len(samp_neighs)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i] - {nodes[i]}) for i in range(len(samp_neighs))]

        if len(pre_hidden_embs) == len(unique_node_list):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_node_list)]
        mask = torch.zeros(len(samp_neighs), len(unique_node_list))
        column_indices = [unique_nodes_dict[node] for samp_neigh in samp_neighs for node in samp_neigh]
        row_indeces = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indeces, column_indices] = 1

        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh)
            aggregate_feats = mask.mm(embed_matrix)

        return aggregate_feats

    def _node_map(self, nodes, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index


class Classification(nn.Module):
    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(emb_size, num_classes)
        )
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        logists = torch.log_softmax(self.layer(embeds), 1)
        return logists
