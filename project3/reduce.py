from __future__ import print_function

from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
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


range_n_components = range(2, Xo.shape[1] + 1)
n_clusters = 2

values = []

for n_components in range_n_components:
    # reducer = PCA(n_components=n_components)
    # reducer = FastICA(n_components=n_components)
    reducer = SparseRandomProjection(n_components=n_components)
    X = reducer.fit_transform(Xo)
    # kur = kurtosis(X)
    # print(kur)
    # values.append(np.absolute(kur).mean())

    # print(mutual_info_classif(X, y))
#endfor

# values = reducer.explained_variance_
# print(values)

# fig = plt.figure()

# plt.plot(range_n_components, values[:-1])
# plt.title("Eigenvalues by Number of Attributes for {} data set".format(fn))
# plt.ylabel("Eigenvalue")
# plt.xlabel("Attributes")

# plt.plot(range_n_components, values)
# plt.title("Avg Kurtosis by Number of Attributes for {} data set".format(fn))
# plt.ylabel("Avg. Kurtosis")
# plt.xlabel("Attributes")

# plt.show()
