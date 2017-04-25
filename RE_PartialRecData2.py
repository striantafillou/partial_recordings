
# coding: utf-8

# In[11]:

def RE_PartialRecData2(hLayerOuts, oLayerOuts, nLayerNeurons, nRecordings, nSamples):
    import numpy as np
    # returns a data set where, nLayerNeurons are recorded from a hidden layer, for each of nRecordings.
    # Each partial recording has nSamples.
    layerArray = np.zeros((nRecordings, nLayerNeurons), dtype=int)
    for iRec in range(nRecordings):
        layerArray[iRec, :]= np.random.choice(range(hLayerOuts.shape[1]), size=nLayerNeurons, replace=True)      

    #print(layerArray)
    X = np.nan*np.zeros((nRecordings*nSamples, hLayerOuts.shape[1]))
    Y = np.nan*np.zeros((nRecordings*nSamples, oLayerOuts.shape[1]))

    # get the data
    sample_ind=0
    for iRec in range(nRecordings):
        rec_inds = np.random.choice(range(hLayerOuts.shape[0]), size=nSamples, replace=True)
        #print(rec_inds)
        cols = layerArray[iRec, :]
        cols = cols[:, None]
        inds = np.array(range(sample_ind,((iRec+1)*nSamples)), dtype=int)
        X[inds, cols] = hLayerOuts[rec_inds, cols]
        Y[inds, :] = oLayerOuts[rec_inds, :]
        sample_ind = sample_ind + nSamples
    return X, Y;

