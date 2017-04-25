
# coding: utf-8

# In[ ]:

def RE_PartialRecData(layer_outputs, nLayerNeurons, nRecordings, nSamples):
    import numpy as np
    # returns a data set where, from each layer, nLayerNeurons[i] neurons are recorded from each layer, for each of nRecordings.
    # Each partial recording has nSamples.
    
    oLayer = len(layer_outputs)-1
    layerNeurons = list()
    # randomly permute samples
    totalNSamples = layer_outputs[0].shape[0]
    random_inds  = np.random.choice(range(totalNSamples), size=totalNSamples, replace=False)
    #print(random_inds[1])
    # choose observed neurons for each layer
    for iLayer in range(len(layer_outputs)):
        #print(iLayer)
        layerArray = np.zeros((nRecordings, nLayerNeurons[iLayer]))
        for iRec in range(nRecordings):
            layerArray[iRec, :]= np.random.choice(range(layer_outputs[iLayer].shape[1]), size=nLayerNeurons[iLayer], replace=True)      
        layerNeurons.append(layerArray)   

    # get the data
    for iLayer in range(len(layer_outputs)):
        if nLayerNeurons[iLayer]==0:
            continue
        sample_ind =0;
        #permuted data
        X_l = layer_outputs[iLayer][random_inds, :]
        for iRec in range(nRecordings):
            notObserved = np.setdiff1d(range(layer_outputs[iLayer].shape[1]), layerNeurons[iLayer][iRec, :])
            inds = np.array(range(sample_ind,((iRec+1)*nSamples)))
            X_l[inds[:, None], notObserved] = np.nan;
            sample_ind = sample_ind + nSamples
        if np.sum(nLayerNeurons[0:iLayer])==0:
            X=X_l
        elif iLayer<oLayer:
            X = np.append(X, X_l, axis=1)
        else:
            Y=X_l
        
    X=X[0:sample_ind, :]
    Y=Y[0:sample_ind, :]
    return X, Y;

