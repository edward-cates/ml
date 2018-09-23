from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from util.graph import graph_feature

def test_svm():
  graph_feature("svm/linear_max_iter", (100, 2000), [
    { "name": "2 Features", "learner": LinearSVC(max_iter=200) },
    # { "name": "10 Features", "learner": DecisionTreeClassifier(min_samples_leaf=2, max_features=6) },
    { "name": "20 Features", "learner": LinearSVC(max_iter=2000) },
  ])
  graph_feature("svm/rbf_max_iter", (100, 2000), [
    { "name": "2 Features", "learner": SVC(max_iter=200) },
    # { "name": "10 Features", "learner": DecisionTreeClassifier(min_samples_leaf=2, max_features=6) },
    { "name": "20 Features", "learner": SVC(max_iter=2000) },
  ])
#endfor
