from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from util.graph import graph_feature

def test_svm():
  graph_feature("svm/max_iter", (1, 500), 10000, lambda n: LinearSVC(max_iter=n))
#endfor
