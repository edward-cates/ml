from sklearn.tree import DecisionTreeClassifier
from util.graph import graph_feature

def test_decision_tree():
  graph_feature("decision_tree/min_samples_leaf", (1, 100), 10000, lambda n: DecisionTreeClassifier(min_samples_leaf=n))
  graph_feature("decision_tree/min_samples_split", (2, 100), 10000, lambda n: DecisionTreeClassifier(min_samples_split=n))
#endfor
