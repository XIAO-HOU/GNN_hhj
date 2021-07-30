import math
import torch

from utils import evaluate
from dataset import Dataset
from models import GraphSage, Classification
from sklearn.utils import shuffle
from torch import nn

cora = Dataset('../data/cora', 'cora')
features, labels, adj_lists, train_ids, val_ids, test_ids = cora.load_data()
features = torch.FloatTensor(features)

num_layers = 2
hidden_emb_size = 128
b_sz = 20

graphSage = GraphSage(num_layers, features.shape[1], hidden_emb_size, features, adj_lists)

num_labels = len(set(labels))
classification = Classification(hidden_emb_size, num_labels)


def train(epoch):
    print('------------------------Epoch %d-------------------------' % epoch)
    train_nodes = shuffle(train_ids)
    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.SGD(params, lr=0.7)
    optimizer.zero_grad()

    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)

    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index*b_sz: (index+1)*b_sz]
        visited_nodes |= set(nodes_batch)
        labels_batch = labels[nodes_batch]

        embs_batch = graphSage(nodes_batch)

        logists = classification(embs_batch)
        loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
        loss /= len(nodes_batch)
        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}]'.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))

        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()


if __name__ == '__main__':
    epochs = 20
    for epoch in range(epochs):
        train(epoch)
        evaluate(val_ids, test_ids, labels, graphSage, classification)
    print('Finished!')
