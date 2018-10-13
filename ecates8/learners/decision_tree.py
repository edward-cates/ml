from sklearn.tree import DecisionTreeClassifier
from util.graph import graph_feature

def test_decision_tree():
  graph_feature("decision_tree/min_samples_leaf", (100, 2000), [
    { "name": "1 Leaf Sample", "learner": DecisionTreeClassifier(min_samples_leaf=1) },
    { "name": "10 Leaf Samples", "learner": DecisionTreeClassifier(min_samples_leaf=10) },
  ])
  graph_feature("decision_tree/max_depth", (100, 2000), [
    { "name": "Max Depth 2", "learner": DecisionTreeClassifier(max_depth=2) },
    { "name": "Max Depth 10", "learner": DecisionTreeClassifier(max_depth=10) },
  ])
  graph_feature("decision_tree/max_features", (100, 2000), [
    { "name": "Max 2 Features", "learner": DecisionTreeClassifier(min_samples_leaf=2, max_features=2) },
    { "name": "Max 10 Features", "learner": DecisionTreeClassifier(min_samples_leaf=2, max_features=10) },
  ])
#endfor
