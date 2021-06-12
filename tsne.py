# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:09:27 2021

@author: Li
"""
#input: numpy array
# what to do?tsne
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
 
def tsne_plot(bases: np.array, extras: np.array):
    #plot the positions of features.
    #:param: bases: parameters of base classifiers or mean feature of base classes
    #:param: extras: parameters of extra classifiers or mean feature of extra classes
    assert(bases.shape[1]==extras.shape[1])
    base_num =  bases.shape[0]
    extra_nums = extras.shape[0]
    total = np.concatenate([bases, extras], axis=0)
    embeded =  TSNE(n_components=2).fit_transform(total)
    plt.scatter([i[0] for i in embeded[:base_num]],[i[1] for i in embeded[:base_num]], c = 'red')
    plt.scatter([i[0] for i in embeded[base_num:]],[i[1] for i in embeded[base_num:]], c = 'blue')
    plt.show()
    
w_1 =np.load('V.npy')
w_2 =np.load('v_trans.npy')
tsne_plot(w_1, w_2)