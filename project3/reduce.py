from __future__ import print_function

from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from sklearn.random_projection import SparseRandomProjection

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
  print(X.shape)

  print(data[:, 4:-1].shape)
  X = np.append(data[:, 4:-1], X, axis=1)
  X = np.append(X, data[:, :1], axis=1)
  print(X.shape)
else:
  X = data[:, :-1]
  print(X.shape)
#endif

Xo = X
y = data[:, -1]


range_n_components = range(2, Xo.shape[1] + 1)
n_clusters = 2
values = []

for n_components in range_n_components:
    # Create a subplot with 1 row and 2 columns
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    # ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    # ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    # reducer = PCA(n_components=n_components)
    reducer = FastICA(n_components=n_components)
    # reducer = SparseRandomProjection(n_components=n_components)
    X = reducer.fit_transform(Xo)
    print(kurtosis(X))
    # print(mutual_info_classif(X, y))

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = GaussianMixture(n_components=n_clusters, random_state=10)
    # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    log_prob, aic, bic = clusterer.score(X), clusterer.aic(X), clusterer.bic(X)
    print("The average log_prob is:", log_prob)
    print("The aic is:", aic)
    print("The bic is:", bic)

    values.append([silhouette_avg, log_prob, aic, bic])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    # sample_silhouette_values = clusterer.score_samples(X)

    # y_lower = 10
    # for i in range(n_clusters):
    #     # Aggregate the silhouette scores for samples belonging to
    #     # cluster i, and sort them
    #     ith_cluster_silhouette_values = \
    #         sample_silhouette_values[cluster_labels == i]

    #     ith_cluster_silhouette_values.sort()

    #     size_cluster_i = ith_cluster_silhouette_values.shape[0]
    #     y_upper = y_lower + size_cluster_i

    #     color = cm.nipy_spectral(float(i) / n_clusters)
    #     ax1.fill_betweenx(np.arange(y_lower, y_upper),
    #                       0, ith_cluster_silhouette_values,
    #                       facecolor=color, edgecolor=color, alpha=0.7)

    #     # Label the silhouette plots with their cluster numbers at the middle
    #     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    #     # Compute the new y_lower for next plot
    #     y_lower = y_upper + 10  # 10 for the 0 samples

    # ax1.set_title("The silhouette plot for the various clusters.")
    # ax1.set_xlabel("The silhouette coefficient values")
    # ax1.set_ylabel("Cluster label")

    # # The vertical line for average silhouette score of all the values
    # ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    # ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # # 2nd Plot showing the actual clusters formed
    # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #             c=colors, edgecolor='k')

    # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # print(centers)
    # Draw white circles at cluster centers
    # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #             c="white", alpha=1, s=200, edgecolor='k')

    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                 s=50, edgecolor='k')

    # ax2.set_title("The visualization of the clustered data.")
    # ax2.set_xlabel("Feature space for the 1st feature")
    # ax2.set_ylabel("Feature space for the 2nd feature")

    # plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
    #               "with n_clusters = %d" % n_clusters),
    #              fontsize=14, fontweight='bold')

values = np.array(values)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.set_title("Silhoutte")
ax1.set_ylim([-1, 1])
ax1.plot(range_n_components, values[:, 0])
ax1.set_xlabel("N Components")

ax2.set_title("Log Likelihood")
ax2.plot(range_n_components, values[:, 1])
ax2.set_xlabel("N Components")

ax3.set_title("AIC / BIC")
ax3.plot(range_n_components, values[:, 2], label="AIC")
ax3.plot(range_n_components, values[:, 3], label="BIC")
ax3.set_xlabel("N Components")
ax3.legend()

plt.show()

print(reducer.explained_variance_)
