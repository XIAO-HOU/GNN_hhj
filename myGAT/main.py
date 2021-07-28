import time

import torch.optim as optim
import torch.nn.functional as F

from dataset import Dataset
from models import GAT
from utils import accuarcy

cora = Dataset('../data/cora', 'cora')
adj, features, labels, idx_train, idx_val, idx_test = cora.load_data()

hidden = 8
dropout = 0.6
nheads = 8
alpha = 0.2
lr = 0.005
weight_decay = 5e-4

model = GAT(nfeat=features.shape[1],
            nhid=hidden,
            nclass=int(labels.max())+1,
            dropout=dropout,
            nheads=nheads,
            alpha=alpha)
optimizer = optim.Adam(model.parameters(),
                       lr = lr,
                       weight_decay=weight_decay)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuarcy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuarcy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuarcy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == '__main__':
    epochs = 100
    for epoch in range(epochs):
        train(epoch)
    test()
