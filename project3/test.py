from __future__ import print_function

from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.random_projection import SparseRandomProjection

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

matplotlib.rcParams.update({'font.size': 22})

fn = sys.argv[1]

data = pd.read_csv('data/{}-dataset.csv'.format(fn)).values

if fn == 'bank':
  X = data[:, 1:4]
  enc = OneHotEncoder(handle_unknown='ignore')
  X = enc.fit_transform(X).toarray()
  print(X.shape)

  print(data[:, 4:-1].shape)
  X = np.append(data[:, 4:-1], X, axis=1)
  X = np.append(X, data[:, :1], axis=1)
  print(X.shape)

  X = normalize(X, axis=0)
else:
  X = data[:, :-1]
  X = normalize(X, axis=0)
  print(X.shape)
#endif

Xo = X
y = data[:, -1]


values = []

n_components = 5
# reducer = PCA(n_components=n_components)
# reducer = FastICA(n_components=n_components)
# reducer = SparseRandomProjection(n_components=n_components)
reducer = RandomTreesEmbedding(n_estimators=n_components, max_depth=3)
X = reducer.fit_transform(Xo).toarray()

# kMeans
x = X

if False:
  n_clusters = 15

  print('clusterer: kMeans')
  print('attributes:', x.shape[1])
  print('clusters:', n_clusters)

  clusterer = KMeans(n_clusters=n_clusters)
  cluster_labels = clusterer.fit_predict(x)
  centers = clusterer.cluster_centers_
  silhouette_avg = silhouette_score(x, cluster_labels)
  print('silhouette:', silhouette_avg)
  sum_sq = 0
  for i in range(n_clusters):
      center = centers[i]
      diff = x[cluster_labels == i] - center
      sq_diff = diff * diff
      sum_sq += sq_diff.sum()
  #endfor
  print('sse:', sum_sq / x.shape[1])
else:
  n_clusters = 18

  print('clusterer: GMM')
  print('attributes:', x.shape[1])
  print('clusters:', n_clusters)

  clusterer = GaussianMixture(n_components=n_clusters)
  cluster_labels = clusterer.fit_predict(x)

  log_prob, aic, bic = clusterer.score(x), clusterer.aic(x), clusterer.bic(x)
  print("The average log_prob is:", log_prob)
  print("The aic is:", aic)
  print("The bic is:", bic)
#endif
