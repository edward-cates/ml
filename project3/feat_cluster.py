from __future__ import print_function

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

class FeatCluster:

  def __init__(self, n_components = 10, n_clusters = 2):
    self.n_components = n_components
    # self.clusterer = KMeans(n_clusters=n_clusters)
    self.clusterer = GaussianMixture(n_components=n_clusters)
  #enddef

  def fit_transform(self, X):
    new_X = []

    for i in range(self.n_components):
      best_score, best_j = -1, 0

      for j in range(X.shape[1]):
        labels = self.clusterer.fit_predict(X)
        # score = silhouette_score(X, labels)
        score = self.clusterer.bic(X)

        if score > best_score:
          best_score, best_j = score, j
        #endif
      #endfor

      new_X.append(labels)
      X = np.append(X[:, 0:j], X[:, j+1:], axis=1)
    #endfor

    return np.array(new_X).transpose()
  #enddef

#endclass
