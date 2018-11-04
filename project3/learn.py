from __future__ import print_function

from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.random_projection import SparseRandomProjection

from feat_cluster import FeatCluster

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

matplotlib.rcParams.update({'font.size': 22})

data = pd.read_csv('data/{}-dataset.csv'.format(sys.argv[1])).values

if False:
  X = data[:, 1:4]
  enc = OneHotEncoder(handle_unknown='ignore')
  X = enc.fit_transform(X).toarray()

  X = np.append(data[:, 4:-1], X, axis=1)
  X = np.append(X, data[:, :1], axis=1)
else:
  X = data[:, :-1]
#endif

print(X.shape)
y = data[:, -1]

reducers = [
  { "label": "FeatCluster", "reducer": FeatCluster },
  { "label": "ICA", "reducer": FastICA },
  { "label": "PCA", "reducer": PCA },
  { "label": "Random", "reducer": SparseRandomProjection },
]

range_n_components = range(2, X.shape[1] + 1, 2)

values = []

for r in reducers:
  for n_components in range_n_components:
    learner = MLPClassifier(solver='sgd', learning_rate='adaptive', max_iter=1000)
    reducer = r["reducer"](n_components=n_components)
    x = reducer.fit_transform(X)
    score = cross_val_score(learner, x, y, cv=10).mean()
    values.append(score)
    print(r["label"], X.shape[1], '=>', n_components, 'components:', score * 100, 'percent correct')
  #endfor
#endfor

n = len(range_n_components)

for i in range(len(reducers)):
  start, end = i*n, (i+1)*n
  plt.plot(range_n_components, values[start:end], label=reducers[i]["label"])
#endfor

plt.show()
