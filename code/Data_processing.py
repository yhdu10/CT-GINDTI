import pandas as pd
import numpy as np
import random

from rdkit import Chem
import networkx as nx
from utils import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    # print(features)
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    # for i in range(c_size):              #add selfloop
    edge_index.append([0, 0])
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index
def seq_cat(prot):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    # print(seq_dict)
    seq_dict_len = len(seq_dict)
    max_seq_len = 1000
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x
def GenerateGraph():
    compound_iso_smiles = []
    Dpath = 'data/drugChe.txt'
    with open(Dpath) as f:
        for line in f.readlines():
            compound_iso_smiles.append(line.strip('\n'))
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        sm = Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=True)
        g = smile_to_graph(sm)
        smile_graph[smile] = g
    return smile_graph
def GeneDTIs(N):
    Dpath = 'data/drugName.txt'
    Tpath = 'data/targetName.txt'

    drugs = []
    with open(Dpath) as f:
        for line in f.readlines():
            drugs.append(line.strip('\n'))

    targets = []
    with open(Tpath) as f:
        for line in f.readlines():
            targets.append(line.strip('\n'))


    rowD = len(drugs)
    colT = len(targets)
    DTIs = np.zeros((rowD,colT))
    count = 0
    while count < N:
        row = random.randint(0, rowD-1)
        col = random.randint(0, colT-1)
        if DTIs[row][col] == 0:
            DTIs[row][col] = 1
            count += 1
    np.savetxt('data/DTIs.npy', DTIs)

def Train_and_TestData():
    train = []
    test = []
    Dpath = 'data/drugName.txt'
    Tpath = 'data/targetName.txt'
    DTIpath = 'data/DTIs.npy'

    drugs = []
    with open(Dpath) as f:
        for line in f.readlines():
            drugs.append(line.strip('\n'))

    prots = []
    with open(Tpath) as f:
        for line in f.readlines():
            prots.append(line.strip('\n'))

    DTIs = np.load(DTIpath, allow_pickle=True)
    index = np.argwhere(DTIs == 1)
    print('sum_edge = ', len(index))

    sumedge = len(index)
    count = int(sumedge / 5)
    while count > 0:
        ind = random.randint(0, sumedge-1)
        row, col = index[ind][0], index[ind][1]
        if DTIs[row][col] == 1 and sum(DTIs[row]) > 1:
            DTIs[row][col] = 0
            test.append(drugs[row] + ',' + prots[col] + ',' + str(row) + ',' + str(col) + ',' + '1')
            count -= 1

    index = np.argwhere(DTIs == 1)
    for ind in index:
        row, col = ind[0], ind[1]
        train.append(drugs[row] + ',' + prots[col] + ',' + str(row) + ',' + str(col) + ',' + '1')
    print('train = ', len(train))
    print('test = ', len(test))
    # print(train)
    # print(test)
    with open('data/trainPost.csv', 'w') as f:
        for tr in train:
            f.write(tr + '\n')
    with open('data/test.csv', 'w') as f:
        for tr in test:
            f.write(tr + '\n')

def TrainPostData():

    postpath = 'data/trainPost.csv'
    Dpath = 'data/drugChe.txt'
    Tpath = 'data/targetSqu.txt'


    drugs = []
    with open(Dpath) as f:
        for line in f.readlines():
            drugs.append(line.strip('\n'))
    prots = []
    with open(Tpath) as f:
        for line in f.readlines():
            prots.append(line.strip('\n'))
    postlist = []
    with open(postpath) as f:
        for line in f.readlines():
            L = line.strip('\n')
            postlist.append(L.split(','))
    print('len(postlist)', len(postlist))
    with open('data/trainPostdata' + '.csv', 'w') as f:
        for t in postlist:
            ls = []
            ls += [drugs[int(t[2])]]
            ls += [prots[int(t[3])]]
            ls += [int(t[4])]
            f.write(','.join(map(str, ls)) + '\n')

def AllTestData():
    testpath = 'data/test.csv'
    Dpath = 'data/drugChe.txt'
    Tpath = 'data/targetSqu.txt'
    DTIpath = 'data/DTIs.npy'

    DTIs = np.load(DTIpath)
    N = sum(sum(DTIs))
    # print(DTIs)
    drugs = []
    with open(Dpath) as f:
        for line in f.readlines():
            drugs.append(line.strip('\n'))
    prots = []
    with open(Tpath) as f:
        for line in f.readlines():
            prots.append(line.strip('\n'))
    testlist = []
    with open(testpath) as f:
        for line in f.readlines():
            L = line.strip('\n')
            testlist.append(L.split(','))

    with open('data/testdata' + '.csv', 'w') as f:
        f.write('compound_iso_smiles,target_sequence,affinity\n')
        for t in testlist:
            ls = []
            ls += [drugs[int(t[2])]]
            ls += [prots[int(t[3])]]
            ls += [int(t[4])]
            f.write(','.join(map(str, ls)) + '\n')

        xlist = []
        r = len(DTIs)
        c = len(DTIs[0])
        count = 0
        for i in range(200000):
            xlist.append(np.random.randint(0, r * c))
        xlist = list(set(xlist))
        for x in xlist:
            x -= 1
            row = int(x / c)
            col = x % c
            if DTIs[row][col] == 0:
                count += 1
                ls = []
                ls += [drugs[row]]
                ls += [prots[col]]
                ls += [0]
                f.write(','.join(map(str, ls)) + '\n')
            while count >= 10*N:
                break
        # for row in range(len(DTIs)):
        #     for col in range(len(DTIs[0])):
        #         if DTIs[row][col] == 0:
        #             ls = []
        #             ls += [drugs[row]]
        #             ls += [prots[col]]
        #             ls += [0]
        #             f.write(','.join(map(str, ls)) + '\n')

def AllTrainData(Negdata):
    postpath = 'data/trainPostdata.csv'
    postlist = []
    with open(postpath) as f:
        for line in f.readlines():
            L = line.strip('\n')
            postlist.append(L)
    print('len(postlist)', len(postlist))
    postlist = postlist + Negdata
    random.shuffle(postlist)
    # print(postlist)
    # print(len(postlist))
    return postlist

def ConToPttestData():
    # convert to PyTorch data format
    smile_graph = GenerateGraph()
    df = pd.read_csv('data/testdata.csv')
    test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
        df['affinity'])
    XT = [seq_cat(t) for t in test_prots]
    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)

    print('preparing test.pt in pytorch format!')
    test_data = TestbedDataset(root='data', dataset='Using_test', xd=test_drugs, xt=test_prots, y=test_Y,
                               smile_graph=smile_graph, operate='w')

def ConToPt2(Neglist):
    Testpath = 'data/testdata.csv'
    testdata = []
    Negdata = []
    with open(Testpath) as f:
        for line in f.readlines():
            testdata.append(line.strip('\n'))
    # print(testdata[0])
    for ni in Neglist:
        Negdata.append(testdata[ni + 1])
    Traindata = AllTrainData(Negdata)
    smile_graph = GenerateGraph()
    train_drugs = []
    train_prots = []
    train_Y = []
    for tr in Traindata:
        tlist = tr.split(',')
        train_drugs.append(tlist[0])
        train_prots.append(tlist[1])
        train_Y.append(int(tlist[2]))
    # print(train_prots[0])
    XT = [seq_cat(t) for t in train_prots]
    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
    print('preparing train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset='Using_train', xd=train_drugs, xt=train_prots,
                                y=train_Y, smile_graph=smile_graph, operate='w')
    return train_data

# GeneTrainTest()
# TrainPostData()
# AllTestData()
# ConToPttestData()

def PredictDTIs(Neglist):
    Testpath = 'data/testdata.csv'
    pNpath = 'data/targetName.txt'
    pSpath = 'data/targetSqu.txt'
    DChepath = 'data/drugChe.txt'
    Dnamepath = 'data/drugName.txt'
    pNlist = []
    pSlist = []
    Dche = []
    Dname = []
    testdata = []
    Predict1 = []
    FinalPredict = []
    with open(Testpath) as f:
        for line in f.readlines():
            testdata.append(line.strip('\n'))
    with open(pNpath) as f:
        for line in f.readlines():
            pNlist.append(line.strip('\n'))
    with open(pSpath) as f:
        for line in f.readlines():
            pSlist.append(line.strip('\n'))
    with open(DChepath) as f:
        for line in f.readlines():
            Dche.append(line.strip('\n'))
    with open(Dnamepath) as f:
        for line in f.readlines():
            Dname.append(line.strip('\n'))
    # print(testdata[0])
    for ni in Neglist:
        Predict1.append(testdata[ni + 1])
    for P in Predict1:
        plist = P.split(',')
        drugIndex = Dche.index(plist[0])
        drugname = Dname[drugIndex]
        PIndex = pSlist.index(plist[1])
        Pname = pNlist[PIndex]
        FinalPredict.append([drugname, Pname, plist[2]])
    return FinalPredict
