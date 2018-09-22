from sklearn.ensemble import AdaBoostClassifier
from util.graph import graph_feature

def test_boosting():
  graph_feature("boosting/ada", (100, 200), 500, lambda n: AdaBoostClassifier())
#endfor
