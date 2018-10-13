from sklearn.neighbors import KNeighborsClassifier
from util.graph import graph_feature

def test_knn():
  graph_feature("k_nearest_neighbors/k", (100, 2000), [
    { "name": "k1", "learner": KNeighborsClassifier(n_jobs=8, n_neighbors=1) },
    { "name": "k8", "learner": KNeighborsClassifier(n_jobs=8, n_neighbors=8) },
  ])
  graph_feature("k_nearest_neighbors/leaf_size", (100, 2000), [
    { "name": "Leaf Size 1", "learner": KNeighborsClassifier(n_jobs=8, leaf_size=1) },
    { "name": "Leaf Size 20", "learner": KNeighborsClassifier(n_jobs=8, leaf_size=20) },
  ])
#endfor
