## Library imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

## Initialize device
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda')


## Hyper-parameters
# bs = 100 ## batch size
# n_latent = 8 ## hidden dimension of RNNs
# population_list = [2,5,4] ## layers 4, 2/3, 5/6 in that order

## Define cortical circuit architecture
class microcircuit(nn.Module):
    def __init__(self, seq_len=1, n_features=64, hidden_dim=8, pop_list = [2,5,4], pEdge=0.3, bsize=100, device=device, manual_seed=0, task='recon', n_classes=None, latticeDim=2):
        super(microcircuit, self).__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.bsize = bsize
        self.device = device
        self.n_features = n_features
        self.pop = pop_list
        self.task = task
        self.n_classes = n_classes
        self.manual_seed = manual_seed
        self.latticeDim = latticeDim

        ##ReLU
        self.relu = nn.ReLU()

        ##RNNs
        self.fullRNN = nn.RNN(input_size = n_features, hidden_size=2*sum(pop_list)*hidden_dim, num_layers = 1, batch_first = True)

        ##Inter-areal connections (backbone)

        self.BBFF = torch.ones((pop_list[1])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBFBa = torch.ones((pop_list[2])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBFBb = torch.ones((pop_list[2])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)

        ##Inter-layer connections (backbone)

        self.BBL423 = torch.ones((pop_list[0])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL2356 = torch.ones((pop_list[1])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL564 = torch.ones((pop_list[2])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)

        self.BBH423 = torch.ones((pop_list[0])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBH2356 = torch.ones((pop_list[1])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBH564 = torch.ones((pop_list[2])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)

        ## Masks (Inter-areal) Shape spec. before .T = (lower,higher)

        self.BBL23H23 = torch.ones((pop_list[1])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL23H56 = torch.ones((pop_list[1])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)

        self.BBL4H4 = torch.ones((pop_list[0])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL4H23 = torch.ones((pop_list[0])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL4H56 = torch.ones((pop_list[0])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)

        self.BBL56H4 = torch.ones((pop_list[2])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL56H23 = torch.ones((pop_list[2])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL56H56 = torch.ones((pop_list[2])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)

        self.BBH4L4 = torch.ones((pop_list[0])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBH4L23 = torch.ones((pop_list[0])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBH4L56 = torch.ones((pop_list[0])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)

        self.BBH23L4 = torch.ones((pop_list[1])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBH23L23 = torch.ones((pop_list[1])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBH23L56 = torch.ones((pop_list[1])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)

        self.BBH56L4 = torch.ones((pop_list[2])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)

        ## Masks (Inter-layer)

        self.BBL456 = torch.ones((pop_list[0])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL234 = torch.ones((pop_list[1])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL5623 = torch.ones((pop_list[2])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)

        self.BBH456 = torch.ones((pop_list[0])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBH234 = torch.ones((pop_list[1])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBH5623 = torch.ones((pop_list[2])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=0,generator=torch.manual_seed(manual_seed)).to(device)

        ## Mask (Input)
        self.BBin = torch.zeros((n_features,2*sum(pop_list)*hidden_dim)).T.to(device)
        self.BBin[:(pop_list[0])*hidden_dim,] = 1.

        ##Output layer to project to original (input) dimension
        if task == 'recon':
            self.opLayer = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=n_features) ##higher 2/3 --> o/p
        elif task == 'classification':
            self.opLayer = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=n_classes)
        elif task == 'lattice':
            self.opLayer = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=n_features)

    def forward(self, x):

        ## Indices for RNNs by layer and area
        startL4 = 0
        endL4 = startL4 + self.pop[0]*self.hidden_dim

        startL23 = endL4
        endL23 = startL23 + self.pop[1]*self.hidden_dim

        startL56 = endL23
        endL56 = startL56 + self.pop[2]*self.hidden_dim

        startH4 = endL56
        endH4 = startH4 + self.pop[0]*self.hidden_dim

        startH23 = endH4
        endH23 = startH23 + self.pop[1]*self.hidden_dim

        startH56 = endH23
        endH56 = startH56 + self.pop[2]*self.hidden_dim

        nSamp, nSteps, inDim = x.shape

        ## Initialize tensor for saving predictions for all steps of the sequence
        if self.task == 'recon':
            pred = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        elif self.task == 'classification':
            pred = torch.zeros(nSamp,nSteps,self.n_classes,requires_grad=False).to(self.device)
        elif self.task == 'lattice':
            pred = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

        ## Initialize hidden states (all random, sampled from [0,1) uniformly)
        h0 = torch.rand(1,nSamp,2*sum(self.pop)*self.hidden_dim,requires_grad=True).to(self.device)

        ## Initialize intermediate outputs
        opL4 = torch.rand(nSamp,nSteps,self.pop[0]*self.hidden_dim).to(self.device)
        opL23 = torch.rand(nSamp,nSteps,self.pop[1]*self.hidden_dim).to(self.device)
        opL56 = torch.rand(nSamp,nSteps,self.pop[2]*self.hidden_dim).to(self.device)

        opH4 = torch.zeros(nSamp,nSteps,self.pop[0]*self.hidden_dim).to(self.device)
        opH23 = torch.zeros(nSamp,nSteps,self.pop[1]*self.hidden_dim).to(self.device)
        opH56 = torch.zeros(nSamp,nSteps,self.pop[2]*self.hidden_dim).to(self.device)

        ## Initialize tensor to store feedforward projections from L23 --> H4
        opProjFF = torch.zeros(nSamp,nSteps,self.pop[0]*self.hidden_dim).to(self.device)

        ## Initialize tensors to store feedback projections from H56 --> L23, L56
        opProjFBa = torch.zeros(nSamp,nSteps,self.pop[1]*self.hidden_dim).to(self.device)
        opProjFBb = torch.zeros(nSamp,nSteps,self.pop[2]*self.hidden_dim).to(self.device)

        ## Initialize matrices for differences between projected feedback and RNN representations
        diffL23 = torch.zeros(nSamp,nSteps,self.pop[1]*self.hidden_dim,requires_grad=False).to(self.device)
        diffL56 = torch.zeros(nSamp,nSteps,self.pop[2]*self.hidden_dim,requires_grad=False).to(self.device)

        ## Steps in for loop
        for ii in range(nSteps):

            ip = torch.unsqueeze(x[:,ii,:],1)

            ## Mask input layer weights
            self.fullRNN._parameters['weight_ih_l0'].data.mul_(self.BBin)

            ## Mask inter-layer weights
            self.fullRNN._parameters['weight_hh_l0'].data[startL4:endL4,startL56:endL56].T.mul_(self.BBL456)
            self.fullRNN._parameters['weight_hh_l0'].data[startL23:endL23,startL4:endL4].T.mul_(self.BBL234)
            self.fullRNN._parameters['weight_hh_l0'].data[startL56:endL56,startL23:endL23].T.mul_(self.BBL5623)

            self.fullRNN._parameters['weight_hh_l0'].data[startH4:endH4,startH56:endH56].T.mul_(self.BBH456)
            self.fullRNN._parameters['weight_hh_l0'].data[startH23:endH23,startH4:endH4].T.mul_(self.BBH234)
            self.fullRNN._parameters['weight_hh_l0'].data[startH56:endH56,startH23:endH23].T.mul_(self.BBH5623)

            ## Mask weights inter-areal weights
            self.fullRNN._parameters['weight_hh_l0'].data[startL23:endL23,startH23:endH23].T.mul_(self.BBL23H23)
            self.fullRNN._parameters['weight_hh_l0'].data[startL23:endL23,startH56:endH56].T.mul_(self.BBL23H56)

            self.fullRNN._parameters['weight_hh_l0'].data[startL4:endL4,startH4:endH4].T.mul_(self.BBL4H4)
            self.fullRNN._parameters['weight_hh_l0'].data[startL4:endL4,startH23:endH23].T.mul_(self.BBL4H23)
            self.fullRNN._parameters['weight_hh_l0'].data[startL4:endL4,startH56:endH56].T.mul_(self.BBL4H56)

            self.fullRNN._parameters['weight_hh_l0'].data[startL56:endL56,startH4:endH4].T.mul_(self.BBL56H4)
            self.fullRNN._parameters['weight_hh_l0'].data[startL56:endL56,startH23:endH23].T.mul_(self.BBL56H23)
            self.fullRNN._parameters['weight_hh_l0'].data[startL56:endL56,startH56:endH56].T.mul_(self.BBL56H56)

            self.fullRNN._parameters['weight_hh_l0'].data[startH4:endH4,startL4:endL4].T.mul_(self.BBH4L4)
            self.fullRNN._parameters['weight_hh_l0'].data[startH4:endH4,startL23:endL23].T.mul_(self.BBH4L23)
            self.fullRNN._parameters['weight_hh_l0'].data[startH4:endH4,startL56:endL56].T.mul_(self.BBH4L56)

            self.fullRNN._parameters['weight_hh_l0'].data[startH23:endH23,startL4:endL4].T.mul_(self.BBH23L4)
            self.fullRNN._parameters['weight_hh_l0'].data[startH23:endH23,startL23:endL23].T.mul_(self.BBH23L23)
            self.fullRNN._parameters['weight_hh_l0'].data[startH23:endH23,startL56:endL56].T.mul_(self.BBH23L56)

            self.fullRNN._parameters['weight_hh_l0'].data[startH56:endH56,startL4:endL4].T.mul_(self.BBH56L4)

            ## Multiply inter-layer backbones
            self.fullRNN._parameters['weight_hh_l0'].data[startL4:endL4,startL23:endL23].T.mul_(self.BBL423)
            self.fullRNN._parameters['weight_hh_l0'].data[startL23:endL23,startL56:endL56].T.mul_(self.BBL2356)
            self.fullRNN._parameters['weight_hh_l0'].data[startL56:endL56,startL4:endL4].T.mul_(self.BBL564)

            self.fullRNN._parameters['weight_hh_l0'].data[startH4:endH4,startH23:endH23].T.mul_(self.BBH423)
            self.fullRNN._parameters['weight_hh_l0'].data[startH23:endH23,startH56:endH56].T.mul_(self.BBH2356)
            self.fullRNN._parameters['weight_hh_l0'].data[startH56:endH56,startH4:endH4].T.mul_(self.BBH564)

            ## Multiply inter-areal backbones
            self.fullRNN._parameters['weight_hh_l0'].data[startL4:endL4,startH23:endH23].mul_(self.BBFF)
            self.fullRNN._parameters['weight_hh_l0'].data[startH56:endH56,startL23:endL23].T.mul_(self.BBFBa)
            self.fullRNN._parameters['weight_hh_l0'].data[startH56:endH56,startL56:endL56].T.mul_(self.BBFBb)


            ## Process data
            xFull, hiddenFull = self.fullRNN(ip,h0)
            h0 = hiddenFull

            opL4[:,ii,:] = torch.squeeze(xFull[:,:,startL4:endL4])
            opL23[:,ii,:] = torch.squeeze(xFull[:,:,startL23:endL23])
            opL56[:,ii,:] = torch.squeeze(xFull[:,:,startL56:endL56])

            opH4[:,ii,:] = torch.squeeze(xFull[:,:,startH4:endH4])
            opH23[:,ii,:] = torch.squeeze(xFull[:,:,startH23:endH23])
            opH56[:,ii,:] = torch.squeeze(xFull[:,:,startH56:endH56])

            ## Final output projection
            pred[:,ii,:] = self.opLayer(torch.squeeze(hiddenFull[:,:,startH23:endH23])) ## H23 --> output

        return pred, (opL23,opL4,opL56,opH23,opH4,opH56), (diffL23, diffL56), (opProjFF,opProjFBa,opProjFBb)

#create the NN
model = microcircuit()
