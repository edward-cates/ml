from sklearn.neighbors import KNeighborsClassifier
from util.graph import graph_feature

def test_knn():
  graph_feature("k_nearest_neighbors/k", (100, 2000), [
    { "name": "k1", "learner": KNeighborsClassifier(n_jobs=8, n_neighbors=1) },
    # { "name": "k5", "learner": KNeighborsClassifier(n_jobs=8, n_neighbors=5) },
    { "name": "k20", "learner": KNeighborsClassifier(n_jobs=8, n_neighbors=8) },
  ])
#endfor
