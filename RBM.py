import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    '''
    This class defines all the functions needed for an RBN model
    activation function : sigmoid
    '''

    def __init__(self,visible_units=256,
                hidden_units = 64,
                k=2,
                learning_rate=1e-3,
                momentum_coefficient=0.5,
                weight_decay = 1e-4,
                use_gpu = True,
                activation='sigmoid'):
        '''
        Defines the model
        W:Wheights shape (visible_units,hidden_units)
        c:hidden unit bias shape (hidden_units , )
        b : visible unit bias shape(visisble_units ,)
        '''
        super(RBM,self).__init__()

        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_gpu = use_gpu
        self.activation = activation

        self.W = torch.randn(self.visible_units,self.hidden_units) / torch.sqrt(self.visible_units)
        self.c = torch.randn(self.hidden_units) / torch.sqrt(self.hidden_units)
        self.b = torch.randn(self.visible_units) / torch.sqrt(self.visible_units)

        self.W_momentum = torch.zeros(self.visible_units,self.hidden_units)
        self.b_momentum = torch.zeros(self.visible_units)
        self.c_momentum = torch.zeros(self.hidden_units)


    def activation(self,X):
        if self.activation=='sigmoid':
            return nn.sigmoid(X)
        elif self.activation=='tanh':
            return nn.tanh(X)
        elif self.activation=='relu':
            return nn.ReLU(X)
        else:
            raise ValueError("Invalid Activation Function")


    def to_hidden(self ,X):
        '''
        Converts the data in visible layer to hidden layer
        also does sampling
        X here is the visible probabilities
        :param X: torch tensor shape = (n_samples , n_features)
        :return -  hidden - new hidden layer (probabilities)
                    sample_h - Gibbs sampling of hidden (1 or 0) based
                                on the value
        '''
        hidden  = self.activation(torch.add(X.dot(self.W),self.c)) #W.x + c
        sample_h = self.sampling(hidden)
        return hidden,sample_h

    def to_visible(self,X):
        '''
        reconstructs data from hidden layer
        also does sampling
        X here is the probabilities in the hidden layer
        :returns - X_dash - the new reconstructed layers(probabilities)
                    sample_X_dash - sample of new layer(Gibbs Sampling)

        '''
        # computing hidden activations and then converting into probabilities
        X_dash = self.activation(torch.add(X.dot(self.W.transpose()) , self.b))
        sample_X_dash = self.sampling(X_dash)
        return X_dash,sample_X_dash

    def sampling(self,s):
        '''does sampling for the change in layer
        Sampling done by Gibbs Sampling useing Bernoulli function
        '''
        s = torch.distributions.Bernoulli(s)
        return s.sample()

    def reconstruction_error(self , data):
        '''
        Computes the reconstruction error for the data
        handled by pytorch by loss functions
        '''
        return contrastive_divergence(data, False)


    def contrastive_divergence(self, input_data ,training = True):
        # positive phase
        positive_hidden_probabilities,positive_hidden_act  = to_hidden(input_data)

        # calculating W via positive side
        positive_association = torch.matmul(input_data.t() , positive_hidden_act)



        # negetive phase
        hidden_activations = positive_hidden_act
        for i in range(self.k):
            visible_p , _ = to_visible(hidden_activations)
            hidden_probabilities,hidden_activations = to_hidden(visible_p)

        negative_visible_probabilities = visible_p
        negative_hidden_probabilities = hidden_probabilities

        # calculating W via negative side
        negative_associations = torch.matmul(negative_visible_probabilities.t() , negative_hidden_probabilities)


        # Update parameters
        if(training):
            self.W_momentum *= self.momentum_coefficient
            self.W_momentum += (positive_associations - negative_associations)

            self.b_momentum *= self.momentum_coefficient
            self.b_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

            self.c_momentum *= self.momentum_coefficient
            self.c_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

            batch_size = input_data.size(0)

            self.weights += self.weights_momentum * self.learning_rate / batch_size
            self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
            self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

            self.weights -= self.weights * self.weight_decay  # L2 weight decay

            # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error


    def forward(self,input_data):
        'data->hidden->visible->hidden'
        return  to_hidden(input_data)
    def backward(self,input_data):
        '''
            Includes the foward prop plus the gradient descent
            Use this for training
        '''
        return contrastive_divergence(input_data , True);


    def train(self,train_data , num_epochs,batch_size):

        if(isinstance(train_data ,torch.utils.data.DataLoader)):
            train_loader = train_data
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)


        for epochs in range(num_epochs):
            epoch_error = 0.0;

            for batch,_ in train_loader:
                batch = batch.view(len(batch) , self.visible_units)

                if(use_gpu):
                    batch = batch.cuda()
                batch_err = self.backward(batch)

                epoch_err += batch_err

            print("Epoch Error (epoch:%d) : %.4f" % (epochs , epoch_err))
        return

    def extract_features(test_dataset):
        if(isinstance(test_data ,torch.utils.data.DataLoader)):
            test_loader = train_data
        else:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

        test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
        test_labels = np.zeros(len(test_dataset))

        for i, (batch, labels) in enumerate(test_loader):
            batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

            if CUDA:
                batch = batch.cuda()

            test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
            test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

        return test_features,test_labels
