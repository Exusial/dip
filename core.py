# -*- coding: utf-8 -*-
"""
Created on Wed May 19 08:48:51 2021

@author: Li
"""
import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader,random_split
from alexnet import alexnet
from dataset import ImageDataset, FeatureDataset
AlexNet = alexnet(pretrained=True)
cls_list = []
data_map = {}

class BaseClassifier(torch.nn.Module):
    def __init__(self, hidden=1000):
        super(BaseClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(4096,hidden)
        self.sigmoid = torch.nn.Sigmoid()
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        return x


def get_w(cls):
    return cls.fc1.weight.cpu().transpose(0,1)


def torch_weight(cls):
    return cls.fc1.weight.cpu().transpose(0,1)


def save_feature(W):
    torch.save(W,"parameters.pt")


def calculate_A():
    '''
    for calculating A.A is a fixed matrix.
    :return: A
    '''
    a_feature = np.zeros((1000,4096))
    x = np.load("base_feature.npy")
    print(x.shape)
    f = open("base_label.txt","r")
    line = f.readlines()
    for i in range(1000):
        idx = int(line[i*100])
        print(idx)
        a_feature[idx-1] = np.mean(x[(idx-1)*100:idx*100,:],0)
    np.save("a_feature.npy",a_feature)
    A = np.zeros((1000,1000))
    for i in range(1000):
        for j in range(1000):
            A[i,j] = np.dot(a_feature[i],a_feature[j]) / (np.linalg.norm(a_feature[i]) * np.linalg.norm(a_feature[j]))
    np.save("A.npy", A)
    return A


def VAGER(W, A, beta=0.1):
    '''VAGER
    :param: W(n*p): parameters of base classifiers (1000*1024*100=100M)
    :param: A(n*n): adjunct matrix (1000*1000=1M)
    :output: V(n*q): embedding of task
    :output: T(q*p): a map from embedding to parameter
    '''
    V =  Variable(torch.rand((1000,400)), requires_grad=True)
    T =  Variable(torch.rand((400,4096)), requires_grad=True)
    optimizer = torch.optim.Adam([V,T],lr=0.1)
    #train using gradient descent(might be more convenient than the analytic method)
    it = 0
    while True:
        loss1 = torch.norm(torch.mm(V,T)-W)
        loss2 = beta*torch.norm(A-torch.mm(V,torch.transpose(V,0,1)))
        loss = loss1+loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        it += 1
        if loss.item()<200 or it > 3000:
            break
    return V,T


def VAGER_v2(W, A, beta=float(0.1)):
    '''VAGER
    :param: W(n*p): parameters of base classifiers (1000*1024*100=100M)
    :param: A(n*n): adjunct matrix (1000*1000=1M)
    :output: V(n*q): embedding of task
    :output: T(q*p): a map from embedding to parameter
    '''
    V = torch.randn(1000,400)
    T = torch.randn(400,4096)
    #train using gradient descent(might be more convenient than the analytic method)
    while True:
        deltaV = 2*(V.mm(T)-W).mm(T.transpose(0,1)) + beta*(-4*A.mm(V)+4*V.mm(V.transpose(0,1)).mm(V))
        V = V-deltaV*1e-4
        deltaT = 2*V.transpose(0, 1).mm(V.mm(T)-W)
        T = T-deltaT*1e-4
        #print(T)
        loss1 = torch.norm(torch.mm(V,T)-W)
        loss2 = beta*torch.norm(A-torch.mm(V,torch.transpose(V,0,1)))
        loss = loss1+loss2
        print(loss)
    return V,T


# VAGER_v2(torch.randn(1000,1024),torch.randn(1000,1000))


def emb_inf(a, V, T):
    '''Embedding inference
    :param: a_new(1*q): center of the new class
    :output: w_new(1*p): parameters of a new classifier
    v_new=a_new(V^t)+
    w_new=v_new T
    '''
    a = a.unsqueeze(0)
    Vt = V.transpose(0,1)
    print(Vt.size())
    print(a.size())
    vnew = torch.mm(a, torch.linalg.pinv(Vt))
    wnew =  torch.mm(vnew,T)
    return wnew


def pretrain():
    # wait for data loading
    data_source = np.load("base_feature.npy")
    f = open("base_label.txt","r")
    line = f.readlines()
    classifier = BaseClassifier(1000).cuda()
    CEloss = torch.nn.CrossEntropyLoss()
    indice = []
    for i in range(1000):
        indice.append(int(line[i*100])-1)
    train_dataset = FeatureDataset(data_source,indice)
    train_dataset.build_dataset()
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0)
    # for test performance
    tlen = int(0.8 * len(train_dataset))
    dlen = len(train_dataset) - tlen
    ts, ds = random_split(train_dataset,(tlen,dlen))
    tl = DataLoader(dataset=ts, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0)
    dl = DataLoader(dataset=ds, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0)
    optimizer = torch.optim.Adam(classifier.parameters(),lr=1e-3)
    epoch = 60
    for _ in range(epoch):
        classifier.train()
        for inp,label in tl:
            output = classifier(inp.cuda())
            loss = CEloss(output,label.cuda())
            loss.backward()
            # print(loss)
            optimizer.step()
        classifier.eval()
        acc = 0
        for inp,label in dl:
            output = classifier(inp.cuda()).cpu()
            for i in range(batch_size):
                if torch.argmax(output[i]) == label[i]:
                    acc += 1
        print(acc / (len(train_dataset) - tlen))
    save_feature(classifier.fc1.weight.cpu())

def tuning_refine(output, y, w, w_tran):
    """
    use sgd
    :param output: model output f
    :param y: label
    :param w: w_n
    :param w_trans: w_N calculated
    :return: common loss
    """
    lamda = 0.4
    CEloss = torch.nn.CrossEntropyLoss()
    pre_loss = CEloss(output,y)
    norm = lamda * torch.norm(w-w_tran).pow(2)
    loss = pre_loss + norm
    return loss


def train():
    # assume all prework is finished
    A = torch.tensor(np.load("A.npy")).to(torch.float32)
    a_feature = np.load("a_feature.npy")
    AlexNet = alexnet(pretrained=True)
    '''
    W = torch.load("parameters.pt")
    W = W.transpose(0,1)
    print(W.shape)
    V, T = VAGER(W, A, beta=float(0.1))
    torch.save(V,"V.pt")
    torch.save(T,"T.pt")
    
    V = torch.load("V.pt")
    T = torch.load("T.pt")

    # use refine for refining w
    # get new average feature
    w_trans = torch.zeros((50, 4096))
    AlexNet.eval()
    for i,folder in enumerate(os.listdir("training")):
        n_feature = torch.zeros((4096))
        f_path = os.path.join("training",folder)
        for img in os.listdir(f_path):
            img_name = os.path.join(f_path,img)
            img = cv2.imread(img_name)
            img = cv2.resize(img, (224, 224))
            img = (img - img.min()) / (img.max() - img.min())
            img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            n_feature += AlexNet(img).squeeze(0)
        n_feature /= 10
        a_n = torch.zeros(1000)
        n_feature = n_feature.detach().numpy()
        print(n_feature)
        for j in range(1000):
            a_n[j] = np.dot(a_feature[i],n_feature) / (np.linalg.norm(a_feature[i]) * np.linalg.norm(n_feature))
        w_trans[i] = emb_inf(a_n,V,T)
    torch.save(w_trans,"w_trans.pt")
    '''
    w_trans = torch.load("w_trans.pt").cuda()
    print(w_trans.size())
    AlexNet = AlexNet.cuda()
    CEloss = torch.nn.CrossEntropyLoss()
    classifier = BaseClassifier(50).cuda() # use preset init.
    train_dataset = ImageDataset()
    train_dataset.build_dataset()
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=16, drop_last=True, num_workers=0)
    optimizer = torch.optim.Adam(classifier.parameters(),lr=1e-2)
    tlen = int(0.8 * len(train_dataset))
    dlen = len(train_dataset) - tlen
    ts, ds = random_split(train_dataset,(tlen,dlen))
    batch_size = 16
    tl = DataLoader(dataset=ts, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0)
    dl = DataLoader(dataset=ds, shuffle=True, batch_size=1, drop_last=True, num_workers=0)
    for _ in range(80):
        classifier.train()
        for img,label in tl:
            i_feature = AlexNet(img.cuda())
            output = classifier(i_feature)
            # print(output)
            nlabel = label.cuda()
            # loss = tuning_refine(output,nlabel,classifier.fc1.weight,w_trans)
            loss = CEloss(output,nlabel)
            loss.backward()
            optimizer.step()
            print(loss)
        classifier.eval()
        acc = 0
        for inp,label in dl:
            output = classifier(AlexNet(inp.cuda())).cpu()
            if torch.argmax(output[0]) == label[0]:
                acc += 1
        print(acc / (len(train_dataset) - tlen))
    infer(AlexNet, classifier)
 

def infer(alex, model):
    # get dataloader
    model.eval()
    md = ImageDataset(data_path='testing')
    train_dataset.build_dataset()
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=1, drop_last=True, num_workers=0)
    acc = 0
    for img,label in train_loader:
        features = alex(img)
        output = model(features)
        pre_label = torch.argmax(output,dim=1,keepdim=False)
        if pre_label == label:
            acc += 1
    print(f"acc is {acc / 2500}")
    model.train()


if __name__ == "__main__":
    # pretrain()
    train()
