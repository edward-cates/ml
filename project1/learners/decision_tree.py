from sklearn.tree import DecisionTreeClassifier
from util.graph import graph_feature

def test_decision_tree():
  graph_feature("decision_tree/min_samples_leaf", (100, 10000), [
    { "name": "Leaf Size 1", "style": "b.-", "learner": DecisionTreeClassifier(min_samples_leaf=1) },
    { "name": "Leaf Size 2", "style": "r.-.", "learner": DecisionTreeClassifier(min_samples_leaf=2) },
    { "name": "Leaf Size 5", "style": "g.--", "learner": DecisionTreeClassifier(min_samples_leaf=5) },
  ])
  # graph_feature("decision_tree/min_samples_split", (2, 100), 10000, lambda n: DecisionTreeClassifier(min_samples_split=n))
#endfor
