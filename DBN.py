import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .RBM import RBM




class DBN(nn.Module):
    def __init__(self,
                n_visible = 256,
                n_hidden = [64 , 100],
                use_cuda = True):
        super(DBN,self).__init()

        self.n_layers = len(n_hidden)
        self.rbm_layers =[]
        self.rbm_nodes = []
        for i in range(self.n_layers ):
            if i==0:
                input_size = n_visible
            else:
                input_size = n_hidden[i-1]
            rbm = RBM(n_visible = input_size, n_hidden = n_hidden[i],use_gpu=use_cuda)

            self.rbm_layers.append(rbm)

        # rbm_layers = [RBM(rbn_nodes[i-1] , rbm_nodes[i],use_gpu=use_cuda) for i in range(1,len(rbm_nodes))]
        self.W_rec = [nn.Parameter(self.rbm_layers[i].W.data.clone()) for i in range(self.n_layers-1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].W.data) for i in range(self.n_layers-1)]
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].c.data.clone()) for i in range(self.n_layers-1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].b.data) for i in range(self.n_layers-1)]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].W.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1]b.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].c.data)

        for i in range(self.n_layers-1):
            self.register_parameter('W_rec%i'%i, self.W_rec[i])
            self.register_parameter('W_gen%i'%i, self.W_gen[i])
            self.register_parameter('bias_rec%i'%i, self.bias_rec[i])
            self.register_parameter('bias_gen%i'%i, self.bias_gen[i])


    def forward(self , input_data):
        '''
            running the forward pass
            do not confuse with training this just runs a foward pass
        '''
        v = input_data
        for i in range(len(rbm_layers)):
            p_v,v = self.rbm_layers[i].forward(v)
        return v

    def train_static(self, train_data,num_epochs,batch_size):
        '''
        Greedy Layer By Layer training
        Keeping previous layers as static
        '''

        v = train_data

        for i in range(len(rbm_layers)):
            print("-"*20)
            print("Training the {} st rbm layer".format(i+1))
            self.rbm_layers[i].train(v , num_epochs,batch_size)
            p_v , v = self.rbm_layers[i].forward(v)


        return

    def train_ith(self, train_data,num_epochs,batch_size,ith_layer):
        '''
        taking ith layer at once
        can be used for fine tuning
        '''
        if(ith_layer>len(rbm_layers)):
            return

        v = train_data
        for ith in range(ith_layer):
            p_v, v = self.rbm_layers[ith].forward(v)


        self.rbm_layers[ith_layer].train(v, num_epochs,batch_size)
        return
