from sklearn.tree import DecisionTreeClassifier
from util.graph import graph_feature

def test_decision_tree():
  graph_feature("decision_tree/min_samples_leaf", (100, 2000), [
    { "name": "Leaf Size 1", "learner": DecisionTreeClassifier(min_samples_leaf=1) },
    # { "name": "Leaf Size 2", "learner": DecisionTreeClassifier(max_depth=10) },
    { "name": "Leaf Size 5", "learner": DecisionTreeClassifier(min_samples_leaf=20) },
  ])
  graph_feature("decision_tree/max_features", (100, 2000), [
    { "name": "2 Features", "learner": DecisionTreeClassifier(min_samples_leaf=2, max_features=2) },
    # { "name": "10 Features", "learner": DecisionTreeClassifier(min_samples_leaf=2, max_features=6) },
    { "name": "20 Features", "learner": DecisionTreeClassifier(min_samples_leaf=2, max_features=10) },
  ])
  # graph_feature("decision_tree/min_samples_split", (2, 100), 10000, lambda n: DecisionTreeClassifier(min_samples_split=n))
#endfor
