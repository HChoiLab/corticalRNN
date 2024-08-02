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
        if task == 'lattice':
            self.L4 = nn.RNN(input_size = n_features+(latticeDim**2)+(pop_list[0]*hidden_dim), hidden_size = pop_list[0]*hidden_dim,
                                 num_layers = 1, batch_first = True)
        else:
            self.L4 = nn.RNN(input_size = n_features+(pop_list[0]*hidden_dim), hidden_size = pop_list[0]*hidden_dim,
                                 num_layers = 1, batch_first = True)

        self.L23 = nn.RNN(input_size = (pop_list[1]+pop_list[1])*hidden_dim, hidden_size = pop_list[1]*hidden_dim,
                                 num_layers = 1, batch_first = True)

        self.L56 = nn.RNN(input_size = (pop_list[2]+pop_list[2])*hidden_dim, hidden_size = pop_list[2]*hidden_dim,
                             num_layers = 1, batch_first = True)

        self.H4 = nn.RNN(input_size = (pop_list[0]+pop_list[0])*hidden_dim, hidden_size = pop_list[0]*hidden_dim,
                             num_layers = 1, batch_first = True)

        self.H23 = nn.RNN(input_size = (pop_list[1])*hidden_dim, hidden_size = pop_list[1]*hidden_dim,
                             num_layers = 1, batch_first = True)

        self.H56 = nn.RNN(input_size = pop_list[2]*hidden_dim, hidden_size = pop_list[2]*hidden_dim,
                             num_layers = 1, batch_first = True)

        ## Intermediate area RNNs
        self.I4 = nn.RNN(input_size = (pop_list[0]+pop_list[0])*hidden_dim, hidden_size = pop_list[0]*hidden_dim,
                             num_layers = 1, batch_first = True)

        self.I23 = nn.RNN(input_size = (pop_list[1]+pop_list[1])*hidden_dim, hidden_size = pop_list[1]*hidden_dim,
                                 num_layers = 1, batch_first = True)

        self.I56 = nn.RNN(input_size = (pop_list[2]+pop_list[2])*hidden_dim, hidden_size = pop_list[2]*hidden_dim,
                             num_layers = 1, batch_first = True)

        ##Inter-areal connections (weights)
        self.WFF1 = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=(pop_list[0])*hidden_dim) ##lower 2/3 --> intermediate 4

        self.WFBa1 = nn.Linear(in_features=(pop_list[2])*hidden_dim, out_features=(pop_list[1])*hidden_dim) ##intermediate 5/6 --> lower 2/3
        self.WFBb1 = nn.Linear(in_features=(pop_list[2])*hidden_dim, out_features=(pop_list[2])*hidden_dim) ##intermediate 5/6 --> lower 5/6

        self.WFF2 = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=(pop_list[0])*hidden_dim) ##intermediate 2/3 --> lower 4

        self.WFBa2 = nn.Linear(in_features=(pop_list[2])*hidden_dim, out_features=(pop_list[1])*hidden_dim) ##higher 5/6 --> intermediate 2/3
        self.WFBb2 = nn.Linear(in_features=(pop_list[2])*hidden_dim, out_features=(pop_list[2])*hidden_dim) ##higher 5/6 --> intermediate 5/6

        ##Inter-layer connections (weights)
        self.WL423 = nn.Linear(in_features=(pop_list[0])*hidden_dim, out_features=(pop_list[1])*hidden_dim)
        self.WL2356 = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=(pop_list[2])*hidden_dim)
        self.WL564 = nn.Linear(in_features=(pop_list[2])*hidden_dim, out_features=(pop_list[0])*hidden_dim)

        self.WI423 = nn.Linear(in_features=(pop_list[0])*hidden_dim, out_features=(pop_list[1])*hidden_dim)
        self.WI2356 = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=(pop_list[2])*hidden_dim)
        self.WI564 = nn.Linear(in_features=(pop_list[2])*hidden_dim, out_features=(pop_list[0])*hidden_dim)

        self.WH423 = nn.Linear(in_features=(pop_list[0])*hidden_dim, out_features=(pop_list[1])*hidden_dim)
        self.WH2356 = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=(pop_list[2])*hidden_dim)
        self.WH564 = nn.Linear(in_features=(pop_list[2])*hidden_dim, out_features=(pop_list[0])*hidden_dim)

        ##Inter-areal connections (backbone)
        self.BBFF1 = torch.ones((pop_list[1])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBFBa1 = torch.ones((pop_list[2])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBFBb1 = torch.ones((pop_list[2])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)

        self.BBFF2 = torch.ones((pop_list[1])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBFBa2 = torch.ones((pop_list[2])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBFBb2 = torch.ones((pop_list[2])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)

        ##Inter-layer connections (backbone)
        self.BBL423 = torch.ones((pop_list[0])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL2356 = torch.ones((pop_list[1])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBL564 = torch.ones((pop_list[2])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)

        self.BBI423 = torch.ones((pop_list[0])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBI2356 = torch.ones((pop_list[1])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBI564 = torch.ones((pop_list[2])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)

        self.BBH423 = torch.ones((pop_list[0])*hidden_dim,(pop_list[1])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBH2356 = torch.ones((pop_list[1])*hidden_dim,(pop_list[2])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)
        self.BBH564 = torch.ones((pop_list[2])*hidden_dim,(pop_list[0])*hidden_dim).T.bernoulli_(p=pEdge,generator=torch.manual_seed(manual_seed)).to(device)

        ##Output layer to project to original (input) dimension
        if task == 'recon':
            self.opLayer = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=n_features) ##higher 2/3 --> o/p
        elif task == 'classification':
            self.opLayer = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=n_classes)
        elif task == 'lattice':
            self.opLayer = nn.Linear(in_features=(pop_list[1])*hidden_dim, out_features=n_features)

        ##Projection initializations (all random)
        self.projL564 = torch.rand(1,seq_len,pop_list[0]*hidden_dim)
        self.projL564 = self.projL564.repeat(self.bsize,1,1).to(device)

        self.projH564 = torch.rand(1,seq_len,pop_list[0]*hidden_dim)
        self.projH564 = self.projH564.repeat(self.bsize,1,1).to(device)

        self.projI564 = torch.rand(1,seq_len,pop_list[0]*hidden_dim)
        self.projI564 = self.projI564.repeat(self.bsize,1,1).to(device)

        ## Projections (inter-areal)
        self.projFBa1 = torch.rand(1,seq_len,pop_list[1]*hidden_dim)
        self.projFBa1 = self.projFBa1.repeat(self.bsize,1,1).to(device)

        self.projFBb1 = torch.rand(1,seq_len,pop_list[2]*hidden_dim)
        self.projFBb1 = self.projFBb1.repeat(self.bsize,1,1).to(device)

        self.projFBa2 = torch.rand(1,seq_len,pop_list[1]*hidden_dim)
        self.projFBa2 = self.projFBa2.repeat(self.bsize,1,1).to(device)

        self.projFBb2 = torch.rand(1,seq_len,pop_list[2]*hidden_dim)
        self.projFBb2 = self.projFBb2.repeat(self.bsize,1,1).to(device)

        ## counter variables
        self.nOld = 0
        self.nNew = 0

    def forward(self, x):

        nSamp, nSteps, inDim = x.shape

        ## Initialize tensor for saving predictions for all steps of the sequence
        if self.task == 'recon':
            pred = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        elif self.task == 'classification':
            pred = torch.zeros(nSamp,nSteps,self.n_classes,requires_grad=False).to(self.device)
        elif self.task == 'lattice':
            pred = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

        ## Initialize hidden states (all random, sampled from [0,1) uniformly)
        h0L4 = torch.rand(1,nSamp,self.pop[0]*self.hidden_dim,requires_grad=True).to(self.device)
        h0L23 = torch.rand(1,nSamp,self.pop[1]*self.hidden_dim,requires_grad=True).to(self.device)
        h0L56 = torch.rand(1,nSamp,self.pop[2]*self.hidden_dim,requires_grad=True).to(self.device)

        h0I4 = torch.rand(1,nSamp,self.pop[0]*self.hidden_dim,requires_grad=True).to(self.device)
        h0I23 = torch.rand(1,nSamp,self.pop[1]*self.hidden_dim,requires_grad=True).to(self.device)
        h0I56 = torch.rand(1,nSamp,self.pop[2]*self.hidden_dim,requires_grad=True).to(self.device)

        h0H4 = torch.rand(1,nSamp,self.pop[0]*self.hidden_dim,requires_grad=True).to(self.device)
        h0H23 = torch.rand(1,nSamp,self.pop[1]*self.hidden_dim,requires_grad=True).to(self.device)
        h0H56 = torch.rand(1,nSamp,self.pop[2]*self.hidden_dim,requires_grad=True).to(self.device)

        ## Initialize intermediate outputs
        opL4 = torch.rand(nSamp,nSteps,self.pop[0]*self.hidden_dim).to(self.device)
        opL23 = torch.rand(nSamp,nSteps,self.pop[1]*self.hidden_dim).to(self.device)
        opL56 = torch.rand(nSamp,nSteps,self.pop[2]*self.hidden_dim).to(self.device)

        opI4 = torch.rand(nSamp,nSteps,self.pop[0]*self.hidden_dim).to(self.device)
        opI23 = torch.rand(nSamp,nSteps,self.pop[1]*self.hidden_dim).to(self.device)
        opI56 = torch.rand(nSamp,nSteps,self.pop[2]*self.hidden_dim).to(self.device)

        opH4 = torch.zeros(nSamp,nSteps,self.pop[0]*self.hidden_dim).to(self.device)
        opH23 = torch.zeros(nSamp,nSteps,self.pop[1]*self.hidden_dim).to(self.device)
        opH56 = torch.zeros(nSamp,nSteps,self.pop[2]*self.hidden_dim).to(self.device)

        ## Initialize tensor to store feedforward projections from L23 --> H4
        opProjFF1 = torch.zeros(nSamp,nSteps,self.pop[0]*self.hidden_dim).to(self.device)
        opProjFF2 = torch.zeros(nSamp,nSteps,self.pop[0]*self.hidden_dim).to(self.device)

        ## Initialize tensors to store feedback projections from H56 --> L23, L56
        opProjFBa1 = torch.zeros(nSamp,nSteps,self.pop[1]*self.hidden_dim).to(self.device)
        opProjFBb1 = torch.zeros(nSamp,nSteps,self.pop[2]*self.hidden_dim).to(self.device)

        opProjFBa2 = torch.zeros(nSamp,nSteps,self.pop[1]*self.hidden_dim).to(self.device)
        opProjFBb2 = torch.zeros(nSamp,nSteps,self.pop[2]*self.hidden_dim).to(self.device)

        ## Initialize temp variables to hold inter-areal feedback
        tempFBa1 = torch.rand(nSamp,self.seq_len,self.pop[1]*self.hidden_dim).to(self.device)
        tempFBb1 = torch.rand(nSamp,self.seq_len,self.pop[2]*self.hidden_dim).to(self.device)

        tempFBa2 = torch.rand(nSamp,self.seq_len,self.pop[1]*self.hidden_dim).to(self.device)
        tempFBb2 = torch.rand(nSamp,self.seq_len,self.pop[2]*self.hidden_dim).to(self.device)

        ## Initialize matrices for differences between projected feedback and RNN representations
        diffL23 = torch.zeros(nSamp,nSteps,self.pop[1]*self.hidden_dim,requires_grad=False).to(self.device)
        diffL56 = torch.zeros(nSamp,nSteps,self.pop[2]*self.hidden_dim,requires_grad=False).to(self.device)

        diffI23 = torch.zeros(nSamp,nSteps,self.pop[1]*self.hidden_dim,requires_grad=False).to(self.device)
        diffI56 = torch.zeros(nSamp,nSteps,self.pop[2]*self.hidden_dim,requires_grad=False).to(self.device)

        ## Steps in for loop
        for ii in range(nSteps):

            ip = torch.unsqueeze(x[:,ii,:],1)

            ## Lower cortical area processing
            concatL4 = torch.cat((ip,self.projL564),dim=2)
            xL4, hiddenL4 = self.L4(concatL4,h0L4) ## i/p + projL564 --> L4
            h0L4 = hiddenL4 ## update hidden state
            opL4[:,ii,:] = torch.squeeze(xL4)
            self.WL423.weight.data.mul_(self.BBL423) ## Mask the weights
            projL423 = self.WL423(xL4) ## L4 --> projL423

            concatL23 = torch.cat((projL423,self.projFBa1),dim=2)
            xL23, hiddenL23 = self.L23(concatL23,h0L23) ## projL423 + projFBa  --> L23
            h0L23 = hiddenL23 ## update hidden state
            opL23[:,ii,:] = torch.squeeze(xL23)
            self.WL2356.weight.data.mul_(self.BBL2356) ## Mask the weights
            projL2356 = self.WL2356(xL23) ## L23 --> projL23
            diffL23[:,ii,:] = opL23[:,ii,:] - torch.squeeze(self.projFBa1) ## Difference between o/p of lower areal RNN and higher areal feedback

            concatL56 = torch.cat((projL2356,self.projFBb1),dim=2)
            xL56, hiddenL56 = self.L56(concatL56,h0L56) ## projL23 + ProjFBb --> L56
            h0L56 = hiddenL56 ## update hidden state
            opL56[:,ii,:] = torch.squeeze(xL56)
            self.WL564.weight.data.mul_(self.BBL564) ## Mask the weights
            projL564 = self.WL564(xL56) ## L56 --> projL56
            diffL56[:,ii,:] = opL56[:,ii,:] - torch.squeeze(self.projFBb1) ## Difference between o/p of lower areal RNN and higher areal feedback

            ## Inter-areal feedforward 1
            self.WFF1.weight.data.mul_(self.BBFF1) ## Mask the weights
            projFF1 = self.WFF1(xL23) ##L23 --> H4
            projFF1 = self.relu(projFF1) ##Only excitatory signals propagate inter-areally
            opProjFF1[:,ii,:] = torch.squeeze(projFF1)

            ## Intermediate cortical area processing
            concatI4 = torch.cat((projFF1,self.projI564),dim=2)
            xI4, hiddenI4 = self.H4(concatI4,h0I4) ## FF + projI564 --> I4
            h0I4 = hiddenI4 ## update hidden state
            opI4[:,ii,:] = torch.squeeze(xI4)
            self.WI423.weight.data.mul_(self.BBI423) ## Mask the weights
            projI423 = self.WI423(xI4) ## I4 --> projI423

            concatI23 = torch.cat((projI423,self.projFBa2),dim=2)
            xI23, hiddenI23 = self.L23(concatI23,h0I23) ## projL423 + projFBa  --> L23
            h0I23 = hiddenI23 ## update hidden state
            opI23[:,ii,:] = torch.squeeze(xI23)
            self.WI2356.weight.data.mul_(self.BBI2356) ## Mask the weights
            projI2356 = self.WI2356(xI23) ## L23 --> projL23
            diffI23[:,ii,:] = opI23[:,ii,:] - torch.squeeze(self.projFBa2) ## Difference between o/p of lower areal RNN and higher areal feedback

            concatI56 = torch.cat((projL2356,self.projFBb1),dim=2)
            xI56, hiddenL56 = self.L56(concatI56,h0I56) ## projL23 + ProjFBb --> L56
            h0I56 = hiddenL56 ## update hidden state
            opI56[:,ii,:] = torch.squeeze(xI56)
            self.WI564.weight.data.mul_(self.BBI564) ## Mask the weights
            projI564 = self.WI564(xI56) ## L56 --> projL56
            diffI56[:,ii,:] = opI56[:,ii,:] - torch.squeeze(self.projFBb2) ## Difference between o/p of lower areal RNN and higher areal feedback

            ## Inter-areal feedforward 2
            self.WFF2.weight.data.mul_(self.BBFF2) ## Mask the weights
            projFF2 = self.WFF2(xI23) ##L23 --> H4
            projFF2 = self.relu(projFF2) ##Only excitatory signals propagate inter-areally
            opProjFF2[:,ii,:] = torch.squeeze(projFF2)

            ## Inter-areal feedback 1
            self.WFBa1.weight.data.mul_(self.BBFBa1) ## Mask the weights
            projFBa1 = self.WFBa1(xI56)
            projFBa1 = self.relu(projFBa1) ##Only excitatory signals propagate inter-areally
            opProjFBa1[:,ii,:] = torch.squeeze(projFBa1)

            self.WFBb1.weight.data.mul_(self.BBFBb1) ## Mask the weights
            projFBb1 = self.WFBb1(xI56)
            projFBb1 = self.relu(projFBb1) ##Only excitatory signals propagate inter-areally
            opProjFBb1[:,ii,:] = torch.squeeze(projFBb1)

            ## Higher cortical area processing
            concatH4 = torch.cat((projFF2,self.projH564),dim=2)
            xH4, hiddenH4 = self.H4(concatH4,h0H4) ## FF + projH564 --> H4
            h0H4 = hiddenH4 ## update hidden state
            opH4[:,ii,:] = torch.squeeze(xH4)
            self.WH423.weight.data.mul_(self.BBH423) ## Mask the weights
            projH423 = self.WH423(xH4) ## H4 --> projH423

            xH23, hiddenH23 = self.H23(projH423,h0H23) ## projH4 --> H23
            h0H23 = hiddenH23 ## update hidden state
            opH23[:,ii,:] = torch.squeeze(xH23)
            projH2356 = self.WH2356(xH23)

            xH56, hiddenH56 = self.H56(projH2356,h0H56) ## projH23 --> H56
            h0H56 = hiddenH56 ## update hidden state
            opH56[:,ii,:] = torch.squeeze(xH56)
            self.WH564.weight.data.mul_(self.BBH564) ## Mask the weights
            projH564 = self.WH564(xH56) ## H56 --> projH56

            ## Inter-areal feedback 2
            self.WFBa2.weight.data.mul_(self.BBFBa2) ## Mask the weights
            projFBa2 = self.WFBa2(xH56)
            projFBa2 = self.relu(projFBa2) ##Only excitatory signals propagate inter-areally
            opProjFBa2[:,ii,:] = torch.squeeze(projFBa2)

            self.WFBb2.weight.data.mul_(self.BBFBb2) ## Mask the weights
            projFBb2 = self.WFBb2(xH56)
            projFBb2 = self.relu(projFBb2) ##Only excitatory signals propagate inter-areally
            opProjFBb2[:,ii,:] = torch.squeeze(projFBb2)

            ## Make updates for the connections starting at L and H 5/6
            with torch.no_grad():

                ## 1-step delay for lateral connections
                self.projL564.copy_(projL564)
                self.projI564.copy_(projI564)
                self.projH564.copy_(projH564)

                ## 2-step time delay for inter-areal feedback connections
                self.projFBa1.copy_(tempFBa1)
                self.projFBb1.copy_(tempFBb1)

                self.projFBa2.copy_(tempFBa2)
                self.projFBb2.copy_(tempFBb2)

                tempFBa1.copy_(projFBa1)
                tempFBb1.copy_(projFBb1)

                tempFBa2.copy_(projFBa2)
                tempFBb2.copy_(projFBb2)

            self.nOld = self.nOld + nSamp
            self.nNew = nSamp

            ## Final output projection
            pred[:,ii,:] = self.opLayer(torch.squeeze(hiddenH23)) ## H23 --> output

        return pred, (opL23,opL4,opL56,opI23,opI4,opI56,opH23,opH4,opH56), (diffL23, diffL56, diffI23, diffI56), (opProjFF1,opProjFBa1,opProjFBb1,opProjFF2,opProjFBa2,opProjFBb2)

#create the NN
model = microcircuit()
