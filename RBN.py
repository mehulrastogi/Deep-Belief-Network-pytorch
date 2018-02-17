import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class RBN(nn.Module):
    '''
    This class defines all the functions needed for an RBN model
    '''
    def __init__(self):
        'Defines the model'
        super(RBN,self).__init__()
        pass


    def forward(self,x):
        'data->hidden->visible->hidden'
        'then the loss and CD and sgd'
        pass

    def to_hidden(self ,X):
        'Converts the data in visible layer to hidden layer'
        'also does sampling'
        pass
    def reconstruct(self,hidden_layer):
        'reconstructs data from hidden layer'
        'also does sampling'
        pass
    def sampling(self,X):
        'does sampling for the change in layer'
        pass
    def free_energy(self,X):
        'does caculation of free energy'
        pass
    def contrastive_divergence(self , x):
        'Computes the Gradients for the updation'
        pass
    def reconstruction_error(self , data):
        'Computes the reconstruction error for the data'
        pass
