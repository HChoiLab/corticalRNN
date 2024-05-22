import numpy as np
from numpy.linalg import norm, eigvals
from numpy.random import choice, random_sample, normal

import copy

from math import factorial
from itertools import permutations

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

"""
Function #1

Generate sequential data that follows the pattern A-B-C-D or A-B-C-C

Parameters:  nTot = Total number of samples to be generated (train + test): int
             fracTrain = Fraction of samples out of nTot to be used for the training set: float in range [0,1]
             nStates = Total number of unique states possible in the sequence: int
             dim = Dimensionality of the data: int
             lenSeq = Length of each complete sequence: int
             pGo = Probability of moving to the next state at the final event of the sequence: float in range [0,1]
             noiseProb = Probability of a sample being noisy (i.e., unclean): float in range [0,1]
             noiseNorm = Magnitude (in the Euclidean sense) of the noise vector by which samples are perturbed: float in range [0,inf), default = 0
             repeats = Whether or not elements of the sequence themselves should be repeated: bool, default = True
             numRepeats = Number of times each element in the sequence is repeated: int, default = 2

Returns:    (repsTrain, yRepsTrain, eTrain), (repsTest, yRepsTest, eTest), (allStates[:nTrain],allStates[nTrain:]), template
            repsTrain =
            yRepsTrain =
            eTrain =
            repsTest = Same as repsTrain, but with nTot*(1-fracTrain) samples for the Test set: ndarray of shape (nTot*(1-fracTrain),dim)
            yRepsTest =
            eTest =
            allStates[:nTrain] =
            allStates[nTrain:] =
            template =
"""

def generate_stay_move(nTot, fracTrain, nStates, dim, lenSeq, pGo, noiseProb, noiseNorm=0, repeats=True, numRepeats=2):

    fracTest = 1 - fracTrain

    nTrain = int(nTot*fracTrain)
    nTest = nTot - nTrain

    ## Generate template representations
    template = np.zeros((nStates, dim))
    for ii in range(nStates):
        template[ii] = choice(2, dim, p=[0.75, 0.25])

    ## Generate set of all actions
    actionSet = choice(2, nTot, p=[1-pGo, pGo])

    ## Starting states for all sequences
    startIdxs = choice(nStates, nTot)

    allStates = np.zeros((nTot,lenSeq),dtype=int)
    allStates[:,0] = startIdxs

    for ii in range(1,lenSeq-1):
        allStates[:,ii] = (allStates[:,ii-1] + 1)%nStates

    ## All sequences as actions
    allStates[:,-1] = (allStates[:,-2] + actionSet)%nStates

    ## Create dataset (all sequences as representations)
    repsDataset = np.zeros((nTot,lenSeq,dim))

    for ii in (range(nTot)):
        repsDataset[ii] = template[allStates[ii]]

    ## Create noise to add to the sequences
    noiseTensor = random_sample(repsDataset.shape)

    ## Change norm of the noise vectors as required
    for kk in range(lenSeq):
        noiseTensor[:,kk,:] = noiseTensor[:,kk,:]/norm(noiseTensor[:,kk,:],axis=-1)[:,None]
    noiseTensor *= noiseNorm

    ## Create noisy representations
    repsDataset = repsDataset + noiseTensor

    ## Repeat sequences if required
    if not repeats:
        numRepeats=1

    repsDataset = np.repeat(repsDataset,numRepeats,axis=1)

    repsTrain = repsDataset[:nTrain,:-numRepeats]
    repsTest = repsDataset[nTrain:,:-numRepeats]

    yRepsTrain = np.expand_dims(repsDataset[:nTrain,-1],1)
    yRepsTest = np.expand_dims(repsDataset[nTrain:,-1],1)

    eTrain = allStates[:nTrain,-1].astype(float)
    eTest = allStates[nTrain:,-1].astype(float)

    return (repsTrain, yRepsTrain, eTrain), (repsTest, yRepsTest, eTest), (allStates[:nTrain],allStates[nTrain:]), template

###################################################################################################
###################################################################################################

"""
Function #2

Generate sequential data that follows the pattern A-B-C-D or A-B-C-X
X is still from the same distribution as the general input stimuli

Parameters:  nTot = Total number of samples to be generated (train + test): int
             fracTrain = Fraction of samples out of nTot to be used for the training set: float in range [0,1]
             nStates = Total number of unique states possible in the sequence: int
             dim = Dimensionality of the data: int
             lenSeq = Length of each complete sequence: int
             pGo = Probability of moving to the next state at the final event of the sequence: float in range [0,1]
             noiseProb = Probability of a sample being noisy (i.e., unclean): float in range [0,1]
             noiseNorm = Magnitude (in the Euclidean sense) of the noise vector by which samples are perturbed: float in range [0,inf), default = 0
             repeats = Whether or not elements of the sequence themselves should be repeated: bool, default = True
             numRepeats = Number of times each element in the sequence is repeated: int, default = 2

Returns:    (repsTrain, yRepsTrain, eTrain), (repsTest, yRepsTest, eTest), (allStates[:nTrain],allStates[nTrain:]), template
            repsTrain =
            yRepsTrain =
            eTrain =
            repsTest = Same as repsTrain, but with nTot*(1-fracTrain) samples for the Test set: ndarray of shape (nTot*(1-fracTrain),dim)
            yRepsTest =
            eTest =
            allStates[:nTrain] =
            allStates[nTrain:] =
            template =
"""

def generate_surprise(nTot, fracTrain, nStates, dim, lenSeq, fracSurprise, noiseNorm=0, repeats=True, numRepeats=2):

    fracTest = 1 - fracTrain

    nTrain = int(nTot*fracTrain)
    nTest = nTot - nTrain

    ## Generate template representations
    template = np.zeros((nStates, dim))
    for ii in range(nStates):
        template[ii] = choice(2, dim, p=[0.75, 0.25])

    ## Generate all sequences -- without any surprises
    startIdxs = choice(nStates, nTot)

    allStates = np.zeros((nTot,lenSeq), dtype=int)
    allStates[:,0] = startIdxs

    lastStates_true = (startIdxs+lenSeq-1)%nStates

    for ii in range(1,lenSeq):
        allStates[:,ii] = (allStates[:,ii-1] + 1)%nStates

    ## Choose indices of samples with "surprises"
    idx_surprise = np.sort(choice(nTot, int(nTot*fracSurprise), replace=False))

    surprise_or_not = np.zeros(nTot, dtype=int)
    for kk in idx_surprise:
        surprise_or_not[kk] = 1

    ## Generate surprise events
    eventSet = choice(nStates, len(idx_surprise))

    ## Replace the last element of the surprise sequences with a "surprise" instead of what is expected
    for kk, idx in enumerate(idx_surprise):
        allStates[idx,-1] = eventSet[kk]

    ## Create Dataset (all sequences as representations)
    repsDataset = np.zeros((nTot,lenSeq,dim))

    for ii in range(nTot):
        repsDataset[ii] = template[allStates[ii]]

    ## Create noise to add to the sequences
    noiseTensor = random_sample(repsDataset.shape)

    ## Change norm of the noise vectors as required
    for kk in range(lenSeq):
        noiseTensor[:,kk,:] = noiseTensor[:,kk,:]/norm(noiseTensor[:,kk,:],axis=-1)[:,None]
    noiseTensor *= noiseNorm

    ## Create noisy representations
    repsDataset = repsDataset + noiseTensor

    ## Repeat sequences if required
    if not repeats:
        numRepeats=1

    repsDataset = np.repeat(repsDataset,numRepeats,axis=1)

    repsTrain = repsDataset[:nTrain,:-numRepeats]
    repsTest = repsDataset[nTrain:,:-numRepeats]

    yRepsTrain = np.expand_dims(repsDataset[:nTrain,-1],1)
    yRepsTest = np.expand_dims(repsDataset[nTrain:,-1],1)

    eTrain = allStates[:nTrain,-1].astype(float)
    eTest = allStates[nTrain:,-1].astype(float)

    ssTrain = surprise_or_not[:nTrain]
    ssTest = surprise_or_not[nTrain:]

    return (repsTrain, yRepsTrain, eTrain, ssTrain), (repsTest, yRepsTest, eTest, ssTest), (allStates[:nTrain],allStates[nTrain:]), template

###################################################################################################
###################################################################################################

"""
Function #3

Generate sequential data where the surprise element could be in any position.
Note: We restrict ourselves to having exactly 1 surprise element (if at all).

Parameters:

Returns:

"""
def generate_violating_sequence(nTot, fracTrain, nStates, dim, lenSeq, fracSurprise, noiseNorm=0, repeats=True, numRepeats=2, template=None):

    fracTest = 1 - fracTrain

    nTrain = int(nTot*fracTrain)
    nTest = nTot - nTrain

    ## Generate template representations
    if template is None:
        ## Generate template representations (for expected elements)
        template = np.zeros((nStates, dim))
        for ii in range(nStates):
            template[ii] = choice(2, dim, p=[0.75, 0.25])
    else:
        template = template

    ## Generate template representations
    # template = np.zeros((nStates, dim))
    # for ii in range(nStates):
    #     template[ii] = choice(2, dim, p=[0.75, 0.25])

    ## Generate all sequences -- without any surprises
    startIdxs = choice(nStates, nTot)

    allStates = np.zeros((nTot,lenSeq), dtype=int)
    allStates[:,0] = startIdxs

    for ii in range(1,lenSeq):
        allStates[:,ii] = (allStates[:,ii-1] + 1)%nStates


    ## Choose indices of samples with "surprises"
    idx_surprise = np.sort(choice(nTot, int(nTot*fracSurprise), replace=False))
    # pos_surprise = choice(np.arange(1,lenSeq-1), len(idx_surprise)) ## control where the surprise element can be placed
    pos_surprise = choice(np.arange(2,3), len(idx_surprise)) ## all surprise elements are in the 2nd position in sequence

    surprise_or_not = np.zeros(nTot, dtype=int) ## binary representation of whether that sequence has a surprise element or not; 1 if surprise
    pos_surprise_arr = np.zeros(nTot, dtype=int) - 1

    for cnt, kk in enumerate(idx_surprise):
        surprise_or_not[kk] = 1
        pos_surprise_arr[kk] = pos_surprise[cnt]

    ## Generate surprise events
    # eventSet = choice(nStates, len(idx_surprise))
    #
    # ## Replace the chosen element of the surprise sequences with a "surprise" instead of what is expected
    # for kk, idx in enumerate(idx_surprise):
    #     pos = pos_surprise[kk]
    #     allStates[idx,pos] = eventSet[kk]

    ## Replace the chosen element of the surprise sequences with the "surprise" instead of what is expected
    for kk, idx in enumerate(idx_surprise):
        pos = pos_surprise[kk]
        allStates[idx,pos] = allStates[idx,pos-1]

    ## Create Dataset (all sequences as representations)
    repsDataset = np.zeros((nTot,lenSeq,dim))

    for ii in range(nTot):
        repsDataset[ii] = template[allStates[ii]]

    ## Create noise to add to the sequences
    noiseTensor = random_sample(repsDataset.shape)

    ## Change norm of the noise vectors as required
    for kk in range(lenSeq):
        noiseTensor[:,kk,:] = noiseTensor[:,kk,:]/norm(noiseTensor[:,kk,:],axis=-1)[:,None]
    noiseTensor *= noiseNorm

    ## Create noisy representations
    repsDataset = repsDataset + noiseTensor

    ## Repeat sequences if required
    if not repeats:
        numRepeats=1

    repsDataset = np.repeat(repsDataset,numRepeats,axis=1)

    repsTrain = repsDataset[:nTrain,:-numRepeats]
    repsTest = repsDataset[nTrain:,:-numRepeats]

    # yRepsTrain = np.expand_dims(repsDataset[:nTrain,-1],1)
    # yRepsTest = np.expand_dims(repsDataset[nTrain:,-1],1)

    ## copied from generate_permuted_sequence
    yRepsTrain = np.zeros_like(repsTrain)
    yRepsTrain[:,1:,:] = repsDataset[:nTrain,(numRepeats+1)*1:]
    yRepsTrain[:,0,:] = yRepsTrain[:,1,:]

    yRepsTest = np.zeros_like(repsTest)
    yRepsTest[:,1:,:] = repsDataset[nTrain:,(numRepeats+1)*1:]
    yRepsTest[:,0,:] = yRepsTest[:,1,:]

    eTrain = allStates[:nTrain,-1].astype(float)
    eTest = allStates[nTrain:,-1].astype(float)

    ssTrain = surprise_or_not[:nTrain]
    ssTest = surprise_or_not[nTrain:]

    posTrain = pos_surprise_arr[:nTrain]
    posTest = pos_surprise_arr[nTrain:]

    return (repsTrain, yRepsTrain, eTrain, posTrain, ssTrain), (repsTest, yRepsTest, eTest, posTest, ssTest), (allStates[:nTrain],allStates[nTrain:]), template

###################################################################################################
###################################################################################################

"""
Function #4

Generate sequential data where the surprise element could be in any position.
Surprise element is different from the regular set of data. We restrict ourselves to having exactly 1 surprise element (if at all)

Parameters:

Returns:

"""

def generate_violate_sequence_surprise_set(nTot, fracTrain, nStates_reg, nStates_surp, dim, lenSeq, fracSurprise, noiseNorm=0, repeats=True, numRepeats=2, template=None, mu=0, sigma=0):

    fracTest = 1 - fracTrain

    nTrain = int(nTot*fracTrain)
    nTest = nTot - nTrain

    if template is None:
        ## Generate template representations (for expected elements)
        template_reg = np.zeros((nStates_reg, dim))
        for ii in range(nStates_reg):
            template_reg[ii] = choice(2, dim, p=[0.75, 0.25])

        ## Generate template representations (for surprise elements)
        template_surp = np.zeros((nStates_surp, dim))
        for ii in range(nStates_surp):
            template_surp[ii] = choice(2, dim, p=[0.75, 0.25]) ##(was 0.15, 0.85)

        template = np.vstack((template_reg, template_surp))
    else:
        template = template

    ## Generate all sequences -- without any surprises
    startIdxs = choice(nStates_reg, nTot)

    allStates = np.zeros((nTot,lenSeq), dtype=int)
    allStates[:,0] = startIdxs

    for ii in range(1,lenSeq):
        allStates[:,ii] = (allStates[:,ii-1] + 1)%nStates_reg

    ## Choose indices of samples with "surprises"
    idx_surprise = np.sort(choice(nTot, int(nTot*fracSurprise), replace=False))
    pos_surprise = choice(np.arange(1,lenSeq-1), len(idx_surprise)) ## control where the surprise element is placed

    surprise_or_not = np.zeros(nTot, dtype=int)
    pos_surprise_arr = np.zeros(nTot, dtype=int) - 1

    for cnt, kk in enumerate(idx_surprise):
        surprise_or_not[kk] = 1
        pos_surprise_arr[kk] = pos_surprise[cnt]

    ## Generate surprise events
    eventSet = choice(nStates_surp, len(idx_surprise)) + nStates_reg

    ## Replace the chosen element of the surprise sequences with a "surprise" instead of what is expected
    for kk, idx in enumerate(idx_surprise):
        pos = pos_surprise[kk]
        allStates[idx,pos] = eventSet[kk]

    ## Create Dataset (all sequences as representations)
    repsDataset = np.zeros((nTot,lenSeq,dim))

    for ii in range(nTot):
        repsDataset[ii] = template[allStates[ii]]

    ## Create noise to add to the sequences
    noiseTensor = random_sample(repsDataset.shape)

    ## Change norm of the noise vectors as required
    for kk in range(lenSeq):
        noiseTensor[:,kk,:] = noiseTensor[:,kk,:]/norm(noiseTensor[:,kk,:],axis=-1)[:,None]
    noiseTensor *= noiseNorm

    ## Create noisy representations
    # repsDataset = repsDataset + noiseTensor
    repsDatasetNoisy = repsDataset + noiseTensor

    ## Generate Gaussian noise tensor
    gaussianNoise = normal(mu,sigma,size=repsDataset.shape)

    ## Add Gaussian noise
    # repsDatasetNoisy = repsDataset + gaussianNoise
    repsDatasetNoisy = repsDatasetNoisy + gaussianNoise

    ## Repeat sequences if required
    if not repeats:
        numRepeats=1

    repsDataset = np.repeat(repsDataset,numRepeats,axis=1)
    repsDatasetNoisy = np.repeat(repsDatasetNoisy,numRepeats,axis=1)

    repsTrain = repsDatasetNoisy[:nTrain,:-numRepeats] ## Noisy train representations
    repsTest = repsDatasetNoisy[nTrain:,:-numRepeats] ## Noisy test representations

    yRepsTrain = repsDataset[:nTrain,(numRepeats+1)*1:]
    yRepsTest = repsDataset[nTrain:,(numRepeats+1)*1:]

    # yRepsTrain = np.expand_dims(repsDataset[:nTrain,-1],1)
    # yRepsTest = np.expand_dims(repsDataset[nTrain:,-1],1)

    eTrain = allStates[:nTrain,-1].astype(float)
    eTest = allStates[nTrain:,-1].astype(float)

    ssTrain = surprise_or_not[:nTrain]
    ssTest = surprise_or_not[nTrain:]

    posTrain = pos_surprise_arr[:nTrain]
    posTest = pos_surprise_arr[nTrain:]

    return (repsTrain, yRepsTrain, eTrain, posTrain, ssTrain), (repsTest, yRepsTest, eTest, posTest, ssTest),(allStates[:nTrain],allStates[nTrain:]), template


###################################################################################################
###################################################################################################

"""
Function #5

Generate sequential data where the last element depends on the overall sequence.

Parameters:

Returns:

"""

def generate_permuted_sequence(nTot, fracTrain, nStates_reg, dim, lenSeq, fracSurprise, noiseNorm=0, repeats=True, numRepeats=2, template=None, mu=0, sigma=0):

    nTrain = nTot
    nStates_surp = factorial(nStates_reg)

    nTrain = int(nTot*fracTrain)
    nTest = nTot - nTrain

    if template is None:
        ## Generate template representations (for n-1 elements)
        template_reg = np.zeros((nStates_reg, dim))
        for ii in range(nStates_reg):
            template_reg[ii] = choice(2, dim, p=[0.75, 0.25])

        ## Generate template representations (for last elements)
        template_surp = np.zeros((nStates_surp, dim))
        for ii in range(nStates_surp):
            template_surp[ii] = choice(2, dim, p=[0.75, 0.25]) ##(was 0.15, 0.85)

        template = np.vstack((template_reg, template_surp))
    else:
        template = template

    ## Generate all sequences -- without any surprises
    all_perm_list = list(permutations(np.arange(nStates_reg)))
    startIdxs = choice(len(all_perm_list), nTot)

    ## Create Dataset (all sequences as representations)
    repsDataset = np.zeros((nTot,lenSeq,dim))
    allStates = np.zeros((nTot,lenSeq), dtype=int)

    for ii in (range(nTot)):
        kk = startIdxs[ii]

        allStates[ii,:-1] = all_perm_list[kk]
        repsDataset[ii,-1] = template[kk+nStates_reg]

        for jj in range(nStates_reg):
            allStates[ii,-1] = kk+nStates_reg
            repsDataset[ii,jj] = template[all_perm_list[kk][jj]]

    ## Choose indices of samples with "surprises"
    idx_surprise = np.sort(choice(nTot, int(nTot*fracSurprise), replace=False))
    pos_surprise = np.ones(len(idx_surprise))*(nStates_reg-1)
    # pos_surprise = choice(np.arange(1,lenSeq-1), len(idx_surprise)) ## control where the surprise element is placed

    surprise_or_not = np.zeros(nTot, dtype=int)
    pos_surprise_arr = np.zeros(nTot, dtype=int) - 1

    for cnt, kk in enumerate(idx_surprise):
        surprise_or_not[kk] = 1
        pos_surprise_arr[kk] = pos_surprise[cnt]

    ## Generate surprise events
    # eventSet = choice(nStates_surp, len(idx_surprise)) + nStates_reg

    ## Replace the chosen element of the surprise sequences with a copy of the previous element instead
    for kk, idx in enumerate(idx_surprise):
        pos = int(pos_surprise[kk])
        allStates[idx,pos] = allStates[idx,pos-1]
        repsDataset[idx,pos] = repsDataset[idx,pos-1]

    ## Create noise to add to the sequences
    noiseTensor = random_sample(repsDataset.shape)

    ## Change norm of the noise vectors as required
    for kk in range(lenSeq):
        noiseTensor[:,kk,:] = noiseTensor[:,kk,:]/norm(noiseTensor[:,kk,:],axis=-1)[:,None]
    noiseTensor *= noiseNorm

    ## Create noisy representations
    # repsDataset = repsDataset + noiseTensor
    repsDatasetNoisy = repsDataset + noiseTensor

    ## Generate Gaussian noise tensor
    gaussianNoise = normal(mu,sigma,size=repsDataset.shape)

    ## Add Gaussian noise
    # repsDatasetNoisy = repsDataset + gaussianNoise
    repsDatasetNoisy = repsDatasetNoisy + gaussianNoise

    ## Repeat sequences if required
    if not repeats:
        numRepeats=1

    repsDataset = np.repeat(repsDataset,numRepeats,axis=1)
    repsDatasetNoisy = np.repeat(repsDatasetNoisy,numRepeats,axis=1)

    repsTrain = repsDatasetNoisy[:nTrain,:-numRepeats] ## Noisy train representations
    repsTest = repsDatasetNoisy[nTrain:,:-numRepeats] ## Noisy test representations

    yRepsTrain = np.zeros_like(repsTrain)
    yRepsTrain[:,1:,:] = repsDataset[:nTrain,(numRepeats+1)*1:]
    yRepsTrain[:,0,:] = yRepsTrain[:,1,:]

    yRepsTest = np.zeros_like(repsTest)
    yRepsTest[:,1:,:] = repsDataset[nTrain:,(numRepeats+1)*1:]
    yRepsTest[:,0,:] = yRepsTest[:,1,:]

    # yRepsTrain = np.expand_dims(repsDataset[:nTrain,-1],1)
    # yRepsTest = np.expand_dims(repsDataset[nTrain:,-1],1)

    eTrain = allStates[:nTrain,2].astype(float)
    eTest = allStates[nTrain:,2].astype(float)

    ssTrain = surprise_or_not[:nTrain]
    ssTest = surprise_or_not[nTrain:]

    posTrain = pos_surprise_arr[:nTrain]
    posTest = pos_surprise_arr[nTrain:]

    return (repsTrain, yRepsTrain, eTrain, posTrain, ssTrain), (repsTest, yRepsTest, eTest, posTest, ssTest),(allStates[:nTrain],allStates[nTrain:]), template


###################################################################################################
###################################################################################################

"""
Function #6

Contextually controlled permuted sequence.

Parameters:

Returns:

"""

def generate_contextually_permuted_sequence(nTot, fracTrain, nStates_reg, dim, lenSeq, fracSurprise=0, noiseNorm=0, repeats=True, numRepeats=2, template=None, mu=0, sigma=0):

    nStates_surp = factorial(nStates_reg)
    nStates_first = nStates_reg

    nTrain = int(nTot*fracTrain)
    nTest = nTot - nTrain

    nTrain_mod = nTot//2
    nTest_mod = nTot//2

    if template is None:
        ## Generate template representations (for n-1 elements)
        template_reg = np.zeros((nStates_reg, dim))
        for ii in range(nStates_reg):
            template_reg[ii] = choice(2, dim, p=[0.75, 0.25])

        ## Generate template representations (for last elements)
        template_surp = np.zeros((nStates_surp, dim))
        for ii in range(nStates_surp):
            template_surp[ii] = choice(2, dim, p=[0.75, 0.25]) ##(was 0.15, 0.85)

        ## Generate template for representations (for 1st elements)
        template_first = np.zeros((nStates_reg, dim))
        for ii in range(nStates_reg):
            template_first[ii] = choice(2, dim, p=[0.75, 0.25])

        template = np.vstack((template_reg, template_surp, template_first))
    else:
        template = template

    ## Generate all sequences -- without any surprises
    all_perm_list_og = list(permutations(np.arange(nStates_reg)))
    all_perm_list_new = all_perm_list_og.copy()

    for cntr in range(len(all_perm_list_og)):
        val = all_perm_list_new[cntr][0]

        inter = list(all_perm_list_new[cntr])

        inter[0] = val + nStates_reg + nStates_surp

        inter_tup = tuple(inter)
        all_perm_list_new[cntr] = inter_tup

    startIdxs = choice(len(all_perm_list_og), nTot)

    repsDataset_og = np.zeros((nTot,lenSeq,dim))
    repsDataset_new = np.zeros((nTot,lenSeq,dim))

    allStates_og = np.zeros((nTot,lenSeq), dtype=int)
    allStates_new = np.zeros((nTot,lenSeq), dtype=int)

    for ii in (range(nTot)):

        kk = startIdxs[ii]

        allStates_og[ii,:-1] = all_perm_list_og[kk]
        allStates_new[ii,:-1] = all_perm_list_new[kk]

        repsDataset_og[ii,-1] = template[kk+nStates_reg]
        repsDataset_new[ii,-1] = template[kk+nStates_reg]

        for jj in range(nStates_reg):
            allStates_og[ii,-1] = kk+nStates_reg
            repsDataset_og[ii,jj] = template[all_perm_list_og[kk][jj]]

            allStates_new[ii,-1] = kk+nStates_reg
            repsDataset_new[ii,jj] = template[all_perm_list_new[kk][jj]]

    ## Choose indices of samples with "surprises"
    idx_surprise = np.zeros(nTot, dtype=np.int8)
    idx_surprise[nTrain_mod:] = 1 ## top half "clean" for both type seqs
    # pos_surprise = np.ones(len(idx_surprise),dtype=np.int8)*(nStates_reg-1)

    surprise_or_not = np.zeros(nTot, dtype=int)
    pos_surprise_arr = np.zeros(nTot, dtype=int) - 1

    for cnt, kk in enumerate(idx_surprise):
        surprise_or_not[cnt] = kk
        if kk:
            pos_surprise_arr[cnt] = nStates_reg-1

    ## Generate surprise events
    # eventSet = choice(nStates_surp, len(idx_surprise)) + nStates_reg

    ## Replace the chosen element of the surprise sequences with a copy of the previous element instead
    for kk, idx in enumerate(idx_surprise):
        pos = int(pos_surprise_arr[kk])

        if idx:
            allStates_og[kk,pos] = allStates_og[kk,pos-1] ## nTot
            repsDataset_og[kk,pos] = repsDataset_og[kk,pos-1]

            allStates_new[kk,pos] = allStates_new[kk,pos-1] ## nTot
            repsDataset_new[kk,pos] = repsDataset_new[kk,pos-1]

    ## Repeat sequences if required
    if not repeats:
        numRepeats=1

    repsDataset_og = np.repeat(repsDataset_og,numRepeats,axis=1) ## size 8
    repsDataset_new = np.repeat(repsDataset_new,numRepeats,axis=1) ## size 8

    # repsDatasetNoisy_og = np.repeat(repsDatasetNoisy_og,numRepeats,axis=1)
    # repsDatasetNoisy_new = np.repeat(repsDatasetNoisy_new,numRepeats,axis=1)

    ##

    repsTrain_og = repsDataset_og[:nTrain_mod,:] ## Noisy train representations
    repsTrain_new = repsDataset_new[:nTrain_mod,:]
    repsClean_all = np.vstack((repsTrain_og,repsTrain_new)) ## same size as nTot

    repsTest_og = repsDataset_og[nTrain_mod:,:] ## Noisy test representations
    repsTest_new = repsDataset_new[nTrain_mod:,:]
    repsSurp_all = np.vstack((repsTest_og,repsTest_new)) ## same size as nTot

    # print(repsClean_all.shape)
    # print(repsSurp_all.shape)

    ## Train and Test sets

    set1_clean_evens = np.where(startIdxs[:nTrain_mod]%2==0)[0]
    set1_clean_odds = np.where(startIdxs[:nTrain_mod]%2==1)[0]

    set2_clean_evens = set1_clean_evens + nTrain_mod
    set2_clean_odds = set1_clean_odds + nTrain_mod

    set1_surp_evens = np.where(startIdxs[nTrain_mod:]%2==0)[0]
    set1_surp_odds = np.where(startIdxs[nTrain_mod:]%2==1)[0]

    set2_surp_evens = set1_surp_evens + nTrain_mod
    set2_surp_odds = set1_surp_odds + nTrain_mod

    reps_new_clean = np.vstack((repsClean_all[set1_clean_evens], repsClean_all[set2_clean_odds],
    repsSurp_all[set1_surp_evens], repsSurp_all[set2_surp_odds]))

    reps_new_surp = np.vstack((repsClean_all[set1_clean_odds], repsClean_all[set2_clean_evens],
    repsSurp_all[set1_surp_odds], repsSurp_all[set2_surp_evens]))

    allStates_new_clean = np.vstack((allStates_og[set1_clean_evens],allStates_new[set1_clean_odds],
    allStates_og[set1_surp_evens],allStates_new[set1_surp_odds]))

    allStates_new_surp = np.vstack((allStates_og[set1_clean_odds],allStates_new[set1_clean_evens],
    allStates_og[set1_surp_odds],allStates_new[set1_surp_evens]))

    ##

    # ids_train = choice(len(reps_new_clean),nTrain,replace=False)
    # ids_test = choice(len(reps_new_surp),nTest,replace=False)

    repsTrain = reps_new_clean#[ids_train]
    repsTest = reps_new_surp#[ids_test]

    # print(repsTrain.shape)
    # print(repsTest.shape)

    allStatesTrain = allStates_new_clean#[ids_train]
    allStatesTest = allStates_new_surp#[ids_test]

    ##

    yRepsTrain = np.zeros((len(repsTrain),(lenSeq-1)*numRepeats,dim))
    yRepsTrain[:,1:,:] = repsTrain[:,(numRepeats+1)*1:]
    yRepsTrain[:,0,:] = yRepsTrain[:,1,:]

    yRepsTest = np.zeros((len(repsTest),(lenSeq-1)*numRepeats,dim))
    yRepsTest[:,1:,:] = repsTest[:,(numRepeats+1)*1:]
    yRepsTest[:,0,:] = yRepsTest[:,1,:]

    ## chop reps
    repsTrain = repsTrain[:,:-numRepeats]
    repsTest = repsTest[:,:-numRepeats]

    eTrain = allStatesTrain[:,2].astype(float)
    eTest = allStatesTest[:,2].astype(float)

    ssTrain = np.zeros(nTrain)
    ssTest = np.ones(nTest)

    posTrain = np.ones(nTrain)*(nStates_reg-1)
    posTest = np.ones(nTest)*(nStates_reg-1)

    return (repsTrain, yRepsTrain, eTrain, posTrain, ssTrain), (repsTest, yRepsTest, eTest, posTest, ssTest),(allStatesTrain,allStatesTest), template

###################################################################################################
###################################################################################################

"""
Function #7

Sequences with a surprise element from another learnt sequnce.
E.g., A-B-C-D becomes A-B-G-D, with E-F-G-H having been one of the other learnt sequences.

Paramteters:
nSeqs = number of sequences
lenSeq = length of each sequence (before any repeats). E.g., A-B-C-D --> 4

Returns:

"""

def generate_known_surprise(nTot, fracTrain, nSeqs, dim, lenSeq, fracSurprise, noiseNorm=0, repeats=True, numRepeats=2, template=None):

    fracTest = 1 - fracTrain

    nTrain = int(nTot*fracTrain)
    nTest = nTot - nTrain

    if template is None:
        ## Generate template representations
        template = np.zeros((nSeqs*lenSeq, dim),dtype=int)
        for ii in range(nSeqs*lenSeq):
            template[ii] = choice(2, dim, p=[0.75, 0.25])
    else:
        template = template

    possible_seqs = np.zeros((nSeqs,lenSeq))
    possible_seqs[0] = np.arange(lenSeq)

    for ii in range(1,nSeqs):
        possible_seqs[ii] = possible_seqs[0] + (ii*lenSeq)

    ## Generate all sequences -- without any surprises
    startIdxs = choice(nSeqs, nTot)

    allStates = np.zeros((nTot,lenSeq), dtype=int)
    for ii in range(nTot):
        allStates[ii] = possible_seqs[startIdxs[ii]]

    ## Choose indices of samples with "surprises"
    idx_surprise = np.sort(choice(nTot, int(nTot*fracSurprise), replace=False))

    surprise_or_not = np.zeros(nTot, dtype=int)
    for kk in idx_surprise:
        surprise_or_not[kk] = 1

    ## Generate surprise events
    pos = 2 ## surprise index
    eventSet = allStates[:,pos].copy()

    ## Replace the "pos" element of the surprise sequences with a "surprise" instead of what is expected
    for kk, ind in enumerate(surprise_or_not):

        if ind:
            starter = startIdxs[kk]
            eventSet[kk] = possible_seqs[starter-1,pos]
            allStates[kk,pos] = eventSet[kk]


    ## Create Dataset (all sequences as representations)
    repsDataset = np.zeros((nTot,lenSeq,dim))

    for ii in range(nTot):
        repsDataset[ii] = template[allStates[ii]]

    ## Create noise to add to the sequences
    noiseTensor = random_sample(repsDataset.shape)

    ## Change norm of the noise vectors as required
    for kk in range(lenSeq):
        noiseTensor[:,kk,:] = noiseTensor[:,kk,:]/norm(noiseTensor[:,kk,:],axis=-1)[:,None]
    noiseTensor *= noiseNorm

    ## Create noisy representations
    repsDataset = repsDataset + noiseTensor

    ## Repeat sequences if required
    if not repeats:
        numRepeats=1

    repsDataset = np.repeat(repsDataset,numRepeats,axis=1)

    repsTrain = repsDataset[:nTrain,:-numRepeats]
    repsTest = repsDataset[nTrain:,:-numRepeats]

    yRepsTrain = np.zeros_like(repsTrain)
    yRepsTest = np.zeros_like(repsTest)

    yRepsTrain[:,:-numRepeats] = repsTrain[:,numRepeats:]
    yRepsTest[:,:-numRepeats] = repsTest[:,numRepeats:]

    yRepsTrain[:,-numRepeats:] = repsDataset[:nTrain,-numRepeats:]
    yRepsTest[:,-numRepeats:] = repsDataset[nTrain:,-numRepeats:]

    eTrain = allStates[:nTrain,pos].astype(float)
    eTest = allStates[nTrain:,pos].astype(float)

    ssTrain = surprise_or_not[:nTrain]
    ssTest = surprise_or_not[nTrain:]

    return (repsTrain, yRepsTrain, eTrain, ssTrain), (repsTest, yRepsTest, eTest, ssTest), (allStates[:nTrain],allStates[nTrain:]), template

###################################################################################################
###################################################################################################

"""
Function #8

Sequences that predict next elements (in binary) of a sequnce depending on the first incoming state.
Temporal violation --> Repeat of the previous state in the 3rd position.

Paramteters:
nSeqs = number of sequences
lenSeq = length of each sequence (before any repeats). E.g., A-B-C-D --> 4

Returns:

"""

def generate_binary_rep_data(nTot, fracTrain, dim, lenSeq, repeats=True, numRepeats=2):

    nTrain = int(nTot*fracTrain)
    nTest = nTot - nTrain

    dec_range = 2**dim - 1
    starterStates = choice(dec_range,nTot,replace=False)
    allStates = np.zeros((nTot,lenSeq),dtype=int)
    repsDataset = np.zeros((nTot,lenSeq,dim),dtype=int)

    for kk in range(lenSeq):
        allStates[:,kk] = (starterStates + kk)%dec_range

    for ii in range(nTot):
        for jj in range(lenSeq):
            x = np.binary_repr(allStates[ii,jj], width=dim)
            for kk, bit in enumerate(x):
                repsDataset[ii,jj,kk] = int(bit)

    repsDataset = np.repeat(repsDataset,numRepeats,axis=1)

    ##Insert temporal violations code here!!

    ## Choose indices of samples with "surprises" in test set
    idx_surprise = np.arange(nTrain,nTot)
    pos_surprise = np.ones(len(idx_surprise))*2

    surprise_or_not = np.zeros(nTot, dtype=int)
    pos_surprise_arr = np.zeros(nTot, dtype=int) - 1

    surprise_or_notTemp = np.zeros(nTot, dtype=int)

    for cnt, kk in enumerate(idx_surprise):
        surprise_or_notTemp[kk] = 1
        pos_surprise_arr[kk] = pos_surprise[cnt]

    ## Copies for temporal violations
    allStatesTemp = np.copy(allStates)
    repsDatasetTemp = np.copy(repsDataset)

    ## Replace the chosen element of the surprise sequences with a copy of the previous element instead
    for kk, idx in enumerate(idx_surprise):
        pos = int(pos_surprise[kk])
        allStatesTemp[idx,pos] = allStatesTemp[idx,pos-1]
        repsDatasetTemp[idx,2*pos:2*pos+2] = repsDatasetTemp[idx,2*pos]

    repsTrain = repsDataset[:nTrain,:-numRepeats]
    repsTest = repsDataset[nTrain:,:-numRepeats]

    repsTestTemp = repsDatasetTemp[nTrain:,:-numRepeats]
    repsTestTemp[:,-numRepeats:] = repsTestTemp[:,2:4]

    yRepsTrain = np.zeros_like(repsTrain)
    yRepsTest = np.zeros_like(repsTest)

    yRepsTrain[:,:-numRepeats] = repsTrain[:,numRepeats:]
    yRepsTest[:,:-numRepeats] = repsTest[:,numRepeats:]

    yRepsTrain[:,-numRepeats:] = repsDataset[:nTrain,-numRepeats:]
    yRepsTest[:,-numRepeats:] = repsDataset[nTrain:,-numRepeats:]

    yRepsTestTemp = np.copy(yRepsTest)
    yRepsTestTemp[:,2:4,:] = yRepsTestTemp[:,0:2,:]

    eTrain = allStates[:nTrain,2].astype(float)
    eTest = allStates[nTrain:,2].astype(float)
    eTestTemp = allStatesTemp[nTrain:,2].astype(float)

    ssTrain = surprise_or_not[:nTrain]
    ssTest = surprise_or_not[nTrain:]
    ssTestTemp = surprise_or_notTemp[nTrain:]

    return (repsTrain, yRepsTrain, eTrain, ssTrain), (repsTest, yRepsTest, eTest, ssTest), (repsTestTemp, yRepsTestTemp, eTestTemp, ssTestTemp), (allStates[:nTrain],allStates[nTrain:],allStatesTemp[nTrain:])

###################################################################################################
###################################################################################################

"""
Function #8

Lattice-based task similar to Recanatesi et al.

Paramteters:
latticeDim = Dimensionality of lattice structure (default 2D)
lenSeq = length of each sequence (before any repeats). E.g., A-B-C-D --> 4

Returns:

"""

def generate_lattice_based_task(nTot, fracTrain, dim, lenSeq, lattice_labels, latticeDim = 2, repeats=True, numRepeats=2, template=None):

    nTrain = int(nTot*fracTrain)
    nTest = nTot - nTrain

    if template is None:
        ## Generate template representations (for expected elements)
        template = np.zeros((nStates, dim))
        for ii in range(nStates):
            template[ii] = choice(2, dim, p=[0.75, 0.25])## p=[Pr(0),Pr(1)]
    else:
        template = template

    binary_moves = choice(2,((lenSeq-1)*nTot)).reshape(nTot,-1)
    move_x = np.where(binary_moves==0)
    move_y = np.where(binary_moves==1)

    actions_x = np.zeros((nTot,lenSeq-1),dtype=int)
    actions_y = np.zeros((nTot,lenSeq-1),dtype=int)
    actions_x[move_x] = choice(np.array([1,-1]),len(move_x[0]))
    actions_y[move_y] = choice(np.array([1,-1]),len(move_y[0]))
    actions = np.dstack((actions_x,actions_y))

    actions_one_hot = np.zeros((nTot,lenSeq-1,latticeDim*2),dtype=int)

    for ii in range(nTot):
        for jj in range(lenSeq-1):
            if actions_x[ii,jj] != 0:
                if actions_x[ii,jj] == 1:
                    actions_one_hot[ii,jj,0:2] = (0,1) ## E = (0100)
                else:
                    actions_one_hot[ii,jj,0:2] = (1,0) ## W = (1000)
            else:
                if actions_y[ii,jj] == 1:
                    actions_one_hot[ii,jj,2:] = (0,1) ## N = (0001)
                else:
                    actions_one_hot[ii,jj,2:] = (1,0) ## S = (0010)

    allStates = np.zeros((nTot,lenSeq,latticeDim),dtype=int)
    allStates[:,0] = choice(np.arange(-2,3),(nTot,latticeDim))
    for kk in range(1,lenSeq):
        allStates[:,kk] = allStates[:,kk-1] + actions[:,kk-1]


    abs_mask = (np.abs(allStates))>2
    sign_mask = (np.sign(abs_mask*allStates))*-5
    allStates += sign_mask

    repsDataset = np.zeros((nTot,lenSeq,dim),dtype=int)

    for ii in range(nTot):
        for jj in range(lenSeq):
            ix = np.where(np.sum(lattice_labels==allStates[ii,jj],1)==2)[0][0]
            repsDataset[ii,jj] = template[ix]

    ##Repeat states
    repsDataset = np.repeat(repsDataset,numRepeats,axis=1)
    actions_one_hot = np.repeat(actions_one_hot,numRepeats,axis=1)

    ##Insert temporal violations code here!!
    ## Choose indices of samples with "surprises" in test set
    idx_surprise = np.arange(nTrain,nTot)
    pos_surprise = np.ones(len(idx_surprise))*2

    surprise_or_not = np.zeros(nTot, dtype=int)
    pos_surprise_arr = np.zeros(nTot, dtype=int) - 1

    surprise_or_notTemp = np.zeros(nTot, dtype=int)

    for cnt, kk in enumerate(idx_surprise):
        surprise_or_notTemp[kk] = 1
        pos_surprise_arr[kk] = pos_surprise[cnt]

    ## Copies for temporal violations
    allStatesTemp = np.copy(allStates)
    repsDatasetTemp = np.copy(repsDataset)
    actions_one_hotTemp = np.copy(actions_one_hot)

    ## Replace the chosen element of the surprise sequences with a copy of the previous element instead
    for kk, idx in enumerate(idx_surprise):
        pos = int(pos_surprise[kk])
        allStatesTemp[idx,pos] = allStatesTemp[idx,pos-1]
        repsDatasetTemp[idx,2*pos:2*pos+2] = repsDatasetTemp[idx,2*pos]
        # actions_one_hotTemp[idx,1*pos:1*pos+2] = 0

    repsTrain = repsDataset[:nTrain,:-numRepeats]
    repsTest = repsDataset[nTrain:,:-numRepeats]

    repsTestTemp = repsDatasetTemp[nTrain:,:-numRepeats]
    repsTestTemp[:,-numRepeats:] = repsTestTemp[:,2:4]

    yRepsTrain = np.zeros_like(repsTrain)
    yRepsTest = np.zeros_like(repsTest)

    yRepsTrain[:,:-numRepeats] = repsTrain[:,numRepeats:]
    yRepsTest[:,:-numRepeats] = repsTest[:,numRepeats:]

    yRepsTrain[:,-numRepeats:] = repsDataset[:nTrain,-numRepeats:]
    yRepsTest[:,-numRepeats:] = repsDataset[nTrain:,-numRepeats:]

    yRepsTestTemp = np.copy(yRepsTest)
    yRepsTestTemp[:,2:4,:] = yRepsTestTemp[:,0:2,:]

    ## actions_one_hot
    actionsTrain = actions_one_hot[:nTrain]
    actionsTest = actions_one_hot[nTrain:]
    actionsTestTemp = actions_one_hotTemp[nTrain:]

    ## Concatenate input reps with actions_one_hot
    repsTrain = np.dstack((repsTrain,actionsTrain))
    repsTest = np.dstack((repsTest,actionsTest))
    repsTestTemp = np.dstack((repsTestTemp,actionsTestTemp))

    ## Same as before
    eTrain = allStates[:nTrain,2].astype(float)
    eTest = allStates[nTrain:,2].astype(float)
    eTestTemp = allStatesTemp[nTrain:,2].astype(float)

    ssTrain = surprise_or_not[:nTrain]
    ssTest = surprise_or_not[nTrain:]
    ssTestTemp = surprise_or_notTemp[nTrain:]

    return (repsTrain, yRepsTrain, eTrain, ssTrain), (repsTest, yRepsTest, eTest, ssTest), (repsTestTemp, yRepsTestTemp, eTestTemp, ssTestTemp), (allStates[:nTrain],allStates[nTrain:],allStatesTemp[nTrain:]), (actionsTrain,actionsTest,actionsTestTemp)


###################################################################################################
###################################################################################################

"""
Create PyTorch dataloaders using numpy data tuples

Parameters:  dataTuple = All information that is to be processed through the dataloader: tuple
             bsize = Batch size: int
             shuffle = Whether or not the data are to be shuffled by the loader: bool

Returns:     loader = Python iterable over the dataset: DataLoader

"""

def makeTensorLoaders(dataTuple,bsize,shuffle):
    ll = len(dataTuple)
    dataTuple = list(dataTuple)

    for ii in range(ll):
        if dataTuple[ii].dtype == 'float64':
            dataTuple[ii] = Variable(torch.from_numpy(dataTuple[ii])).requires_grad_(True)
        else:
            dataTuple[ii] = Variable(torch.from_numpy(dataTuple[ii])).requires_grad_(False)

    dataTuple = tuple(dataTuple)

    tensorData = TensorDataset(*dataTuple)
    loader = DataLoader(tensorData, batch_size=bsize, shuffle=shuffle)

    return loader

###################################################################################################
###################################################################################################
