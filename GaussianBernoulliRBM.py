from RBM import RBM
import torch

class GaussianBernoulliRBM(RBM):

    '''
    Visisble layer can assume real values
    Hidden layer assumes Binarry Values only

    '''

    def to_visible(self,X):
        '''
        the visible units follow gaussian distributions here
        :params X: torch tensor shape = (n_samples , n_features)
        :returns X_prob - the new reconstructed layers(probabilities)
                sample_X_prob - sample of new layer(Gibbs Sampling)

        '''

        X_prob = torch.matmul(X ,self.W.transpose( 0 , 1) )
        X_prob = torch.add(X_prob , self.v_bias)

        sample_X_prob = X_prob + torch.randn(X_prob.shape)

        return X_prob,sample_X_prob
