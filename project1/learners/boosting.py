from sklearn.ensemble import AdaBoostClassifier
from util.graph import graph_feature

def test_boosting():
  graph_feature("boosting/estimators", (100, 2000), [
    { "name": "2 Features", "learner": AdaBoostClassifier(n_estimators=10) },
    { "name": "20 Features", "learner": AdaBoostClassifier(n_estimators=100) },
  ])
  graph_feature("boosting/learning_rate", (100, 2000), [
    { "name": "2 Features", "learner": AdaBoostClassifier(learning_rate=1) },
    { "name": "20 Features", "learner": AdaBoostClassifier(learning_rate=0.5) },
  ])
#endfor
