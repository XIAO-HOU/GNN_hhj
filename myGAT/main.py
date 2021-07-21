import time
import numpy as np
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

    loss_val = F.nll_loss(output[idx_val], labels[idx_train])
    acc_val = accuarcy(output[idx_val], labels[idx_val])


if __name__ == '__main__':
    epochs = 100
    for epoch in range(epochs):
        train(epoch)
