
# coding: utf-8

# In[9]:

def RE_PartialRecData(layer_outputs, nLayerNeurons, nRecordings, nSamples):
    import numpy as np
    # returns a data set where, from each layer, nLayerNeurons[i] neurons are recorded from each layer, for each of nRecordings.
    # Each partial recording has nSamples.
    
    oLayer = len(layer_outputs)-1
    layerNeurons = list()
    for iLayer in range(len(layer_outputs)):
        #print(iLayer)
        layerArray = np.zeros((nRecordings, nLayerNeurons[iLayer]))
        for iRec in range(nRecordings):
            layerArray[iRec, :]= np.random.choice(range(layer_outputs[iLayer].shape[1]), size=nLayerNeurons[iLayer], replace=True)      
        layerNeurons.append(layerArray)   

    if len(np.unique(layerNeurons[oLayer]))<layer_outputs[oLayer].shape[1]:
    #     #pick #outputs random places and replace them with 1:#outputs.
         layerNeurons[3][np.random.choice(range(nRecordings), size =layer_outputs[iLayer].shape[1], replace=False), 0] = range(10)   

    for iLayer in range(len(layer_outputs)):
        if nLayerNeurons[iLayer]==0:
            continue
        sample_ind =0;
        X_l = layer_outputs[iLayer]
        for iRec in range(nRecordings):
            notObserved = np.setdiff1d(range(layer_outputs[iLayer].shape[1]), layerNeurons[iLayer][iRec, :])
            inds = np.array(range(sample_ind,((iRec+1)*nSamples)))
            X_l[inds[:, None], notObserved] = np.nan;
            sample_ind = sample_ind + nSamples
        if np.sum(nLayerNeurons[0:iLayer])==0:
            X=X_l
        else:
            if iLayer==oLayer:
                Y = X_l
            else:
                X = np.append(X, X_l, axis=1)
    return X, Y;

