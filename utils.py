# Temporarily dumping not so useful functions here




def free_energy(self,v):
    '''
    Does caculation of free energy
    '''
    v_bias = v.mv(self.b)
    wx_b = torch.clamp(F.liinear(v,self.W,self.c),-80,80)
    hidden_term  = wx_b.exp().add(1).log().sum(1)
    return(-hidden_term - v_bias).mean()
