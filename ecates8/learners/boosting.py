from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from util.graph import graph_feature

def test_boosting():
  graph_feature("boosting/estimators", (100, 2000), [
    { "name": "4 Estimators", "learner": AdaBoostClassifier(n_estimators=4) },
    { "name": "100 Estimators", "learner": AdaBoostClassifier(n_estimators=100) },
  ])
  graph_feature("boosting/learning_rate", (100, 2000), [
    { "name": "Learning Rate 2", "learner": AdaBoostClassifier(learning_rate=2) },
    { "name": "Learning Rate 0.25", "learner": AdaBoostClassifier(learning_rate=0.25) },
  ])
  graph_feature("boosting/max_features", (100, 2000), [
    { "name": "4 Features", "learner": GradientBoostingClassifier(max_features=4) },
    { "name": "12 Features", "learner": GradientBoostingClassifier(max_features=12) },
  ])
#endfor
