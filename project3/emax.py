from __future__ import print_function

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
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

  range_n_clusters = range(2, 41, 4)
  # range_n_clusters = [15]
  range_n_components = [4, 17, 30]
else:
  X = data[:, :-1]
  X = normalize(X, axis=0)
  range_n_clusters = range(2, 21, 2)
  # range_n_clusters = [2]
  range_n_components = [2, 5, 8, 11, 14]
  print(X.shape)
#endif

y = data[:, -1]

# range_n_components = range(2, X.shape[1] + 1)

# pca = PCA(n_components=2)
# X = pca.fit_transform(X)
# print(X.shape)

fig, (ax2, ax3) = plt.subplots(1, 2)

for n_components in range_n_components:
  values = []

  for n_clusters in range_n_clusters:
    # reducer = PCA(n_components=n_components)
    # reducer = SparseRandomProjection(n_components=n_components)
    # reducer = FastICA(n_components=n_components)
    # x = reducer.fit_transform(X)
    reducer = RandomTreesEmbedding(n_estimators=n_components, max_depth=3)
    x = reducer.fit_transform(X).toarray()
    print(x.shape)

    clusterer = GaussianMixture(n_components=n_clusters)
    cluster_labels = clusterer.fit_predict(x)

    silhouette_avg = 0 #silhouette_score(X, cluster_labels)

    log_prob, aic, bic = clusterer.score(x), clusterer.aic(x), clusterer.bic(x)
    print("The average log_prob is:", log_prob)
    print("The aic is:", aic)
    print("The bic is:", bic)

    values.append([silhouette_avg, log_prob, aic, bic])
  #endfor
  x_label = "N Clusters"

  # range_n_clusters = range_n_components
  # x_label = "Attributes"

  values = np.array(values)
  print(values)

  ax2.set_title("Log Likelihood")
  ax2.plot(range_n_clusters, values[:, 1], label="{} Attr.".format(n_components))
  ax2.set_xlabel(x_label)
  # ax2.set_ylim([0, 150])
  ax2.legend()

  ax3.set_title("BIC")
  # ax3.plot(range_n_clusters, values[:, 2], label="AIC")
  ax3.plot(range_n_clusters, values[:, 3], label="{} Attr.".format(n_components))
  ax3.set_xlabel(x_label)
  # ax3.set_ylim([-15e5, 0])
  ax3.legend()

#endfor


plt.show()
