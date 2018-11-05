# Temporarily dumping not so useful functions here

# def extract_features(test_dataset):
#     if(isinstance(test_data ,torch.utils.data.DataLoader)):
#         test_loader = train_data
#     else:
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
#
#     test_features = np.zeros((len(test_dataset), self.hidden_units))
#     test_labels = np.zeros(len(test_dataset))
#
#     for i, (batch, labels) in enumerate(test_loader):
#         batch = batch.view(len(batch), self.visible_units)  # flatten input data
#
#         if self.use_gpu:
#             batch = batch.cuda()
#
#         test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = self.to_hidden(batch).cpu().numpy()
#         test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()
#
#     return test_features,test_labels


# def free_energy(self,v):
#     '''
#     Does caculation of free energy
#     '''
#     v_bias = v.mv(self.b)
#     wx_b = torch.clamp(F.liinear(v,self.W,self.c),-80,80)
#     hidden_term  = wx_b.exp().add(1).log().sum(1)
#     return(-hidden_term - v_bias).mean()
