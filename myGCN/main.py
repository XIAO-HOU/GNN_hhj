import time

import scipy.sparse as sp
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import GCN
from dataset import Dataset
from utils import normalize, accuracy, sparse_mx_to_torch_sparse_tensor


cora = Dataset('../data/cora', 'cora')
cora.load_data()

features = getattr(cora, 'cora_features')
labels = getattr(cora, 'cora_labels')
# adj_list = getattr(cora, 'cora_adj_list')
adj_matrix = getattr(cora, 'cora_adj_matrix')

train_indexes = getattr(cora, 'cora_train')
val_indexes = getattr(cora, 'cora_val')
test_indexes = getattr(cora, 'cora_test')

features = normalize(features)
adj_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))

features = torch.FloatTensor(features)
labels = torch.LongTensor(labels)
adj_matrix = sparse_mx_to_torch_sparse_tensor(adj_matrix)

hidden = 16
dropout = 0.5
lr = 0.01
weight_decay = 5e-4

model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=labels.max().item() + 1,
            dropout=dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_matrix)
    loss_train = F.nll_loss(output[train_indexes], labels[train_indexes])
    acc_train = accuracy(output[train_indexes], labels[train_indexes])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features, adj_matrix)

    loss_val = F.nll_loss(output[val_indexes], labels[val_indexes])
    acc_val = accuracy(output[val_indexes], labels[val_indexes])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj_matrix)
    loss_test = F.nll_loss(output[test_indexes], labels[test_indexes])
    acc_test = accuracy(output[test_indexes], labels[test_indexes])
    print('Test set results:',
          'loss={:.4f}'.format(loss_test.item()),
          'accuracy={:.4f}'.format(acc_test.item()))


if __name__ == '__main__':
    epochs = 200
    start_time = time.time()
    for epoch in range(epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - start_time))

    test()


