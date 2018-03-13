import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .RBN import RBN




class DBN(nn.Module):
    def __init__(self,
                n_visible = 256,
                n_hidden = [64 , 100]):
        super(DBN,self).__init()

        rbn_nodes =[]
        for i in range(len(n_hidden)):
            if i==0:

                rbn_nodes.append(n_visible)
            else:
                rbn_nodes.append(n_hidden[i-1])

        rbn_layers = [RBN(rbn_nodes[i-1] , rbn_nodes[i]) for i in range(1,len(rbn_nodes))]


        '''
        self.W = []
        self.b = []
        self.c = []
        '''

        pass
    def forward(self , visible):
        '''running the forward pass'''
