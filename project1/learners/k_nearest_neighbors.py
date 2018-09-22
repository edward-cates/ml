from sklearn.neighbors import KNeighborsClassifier
from util.graph import graph_feature

def test_knn():
  graph_feature("k_nearest_neighbors/k", range(1, 40, 1), 3000, lambda n: KNeighborsClassifier(n_jobs=10, n_neighbors=n))
#endfor
