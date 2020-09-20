
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
import logging
import os
import sys



class Simplenet_v2(nn.Module):
    """
    This is Simplenet but with only one small Linear layer, instead of two Linear layers,
    one of which is large.
    26K parameters.
    python compress_classifier.py ${MNIST_PATH} --arch=simplenet_mnist --vs=0 --lr=0.01

    ==> Best [Top1: 98.970   Top5: 99.970   Sparsity:0.00   Params: 26000 on epoch: 54]
    """
    def __init__(self):
        super(Simplenet_v2,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2, 2) # (20,12,12)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2, 2)# (50,4,4)
        self.avgpool = nn.AvgPool2d(4, stride=1) # (50,1,1)
        self.fc = nn.Linear(50, 10) 
        self.softmax = nn.Softmax(dim=1) #(1,10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))# (20,12,12)
        x = self.pool2(self.relu2(self.conv2(x)))# (50,4,4)
        x = self.avgpool(x)# (50,1,1)
        x = x.view(x.size(0), -1) #(50)
        x = self.fc(x) #(10)
        x = self.softmax(x) #(10)

        return x


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()
        # input (1,28,28)
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # (20,24,24)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2) # (20,12,12)
        self.conv2 = nn.Conv2d(20, 50, 5, 1) # (50,8,8)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2) # (50,4,4)
        self.fc1 = nn.Linear(4*4*50, 500) #(1,500)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, 10) #(1,10)
        self.softmax = nn.Softmax(dim=1) #(1,10)

    def forward(self,x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 4*4*50)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x



def profile_for_nn(x,model):
    from thop import profile
    
    macs, params = profile(model, inputs=(x, ))


    return str(macs/1e6)+'M',str(params/1e3)+"K"

if __name__ == '__main__':
    # main()
    from models.resnet import *
    model1 = resnet34()
    model2 = TestNet()
    model3 = Simplenet_v2()
    print('resnet34',profile_for_nn(torch.randn(1,3,224,224),model1))
    print('TestNet',profile_for_nn(torch.randn(1,1,28,28),model2))
    print('Simplenet_v2',profile_for_nn(torch.randn(1,1,28,28),model3))


