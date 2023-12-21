import numpy as np
import pandas as pd
from random import shuffle
import torch
import torch.nn as nn
from model import GINConvNet
from utils import *
from Data_processing import ConToPt2
from Data_processing import PredictDTIs
from sklearn.metrics import roc_auc_score as AUROC


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss1 = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx  == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            loss1 = loss.item()
    return loss1

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

cuda_name = "cuda:0"
print('cuda_name:', cuda_name)
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0001

# training the model
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = GINConvNet.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_mse = 1000
best_epoch = -1
test_data = TestbedDataset(root='data', dataset='Using_test', operate='r')

def Train1(train_data):
    Ranklist = []
    Relist = []
    ROClist = []
    losslist = []
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    for epoch in range(100):
        loss1 = train(model, device, train_loader, optimizer, epoch+1)
        losslist.append(loss1)
        G,P = predicting(model, device, test_loader)
        rank, recall, Neg, Predict_Highest = Evaluation(G, P)
        roc = AUROC(G, P)
        Ranklist.append(rank)
        Relist.append(recall)
        ROClist.append(roc)
    return Ranklist, Relist, ROClist, Neg, Predict_Highest

def Main():
    ResultRanklist = []
    ResultRelist = []
    ResultROClist= []
    Neglist = []
    train_data = ConToPt2(Neglist)
    Rank, Recall, ROC, Neglist, Predict_Highest = Train1(train_data)
    count = 10
    for i in range(count):
        train_data = ConToPt2(Neglist)
        Rank, Recall, ROC, Neglist, Predict_Highest= Train1(train_data)
        ResultRanklist.append(Rank)
        ResultRelist.append(Recall)
        ResultROClist.append(ROC)

    prdict = PredictDTIs(Predict_Highest)
    print(prdict)
    with open('data/DrawGraphEdge.csv', 'w') as f:
        for e in prdict:
            f.write(e[0] + ',' + e[1] + ',' + e[2] + '\n')
    print(ResultRanklist)
    print(ResultRelist)
    print(ResultROClist)

Main()








