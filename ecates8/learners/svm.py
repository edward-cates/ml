from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from util.graph import graph_feature

def test_svm():
  graph_feature("svm/poly_max_iter", (100, 2000), [
    { "name": "1000 Iterations", "learner": SVC(kernel="poly", max_iter=1000) },
    { "name": "4000 Iterations", "learner": SVC(kernel="poly", max_iter=4000) },
  ])
  graph_feature("svm/rbf_max_iter", (100, 2000), [
    { "name": "200 Iterations", "learner": SVC(max_iter=200) },
    { "name": "2000 Iterations", "learner": SVC(max_iter=2000) },
  ])
#endfor
