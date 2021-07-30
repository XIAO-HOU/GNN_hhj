import torch

from sklearn.metrics import f1_score


def evaluate(val_ids, test_ids, labels, graphSage, classification):
    models =  [graphSage, classification]

    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)

    embs = graphSage(val_ids)
    logits = classification(embs)
    _, predicts = torch.max(logits, 1)
    labels_val = labels[val_ids]
    comps = zip(labels_val, predicts.data)

    val_f1 = f1_score(labels_val, predicts.data, average='micro')
    print('Validation F1: ', val_f1)

    embs = graphSage(test_ids)
    logits = classification(embs)
    _, predicts = torch.max(logits, 1)
    labels_val = labels[test_ids]
    comps = zip(labels_val, predicts.data)

    val_f1 = f1_score(labels_val, predicts.data, average='micro')
    print('Test F1: ', val_f1)

    for param in params:
        param.requires_grad = True
