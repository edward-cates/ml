from sklearn.neural_network import MLPClassifier
from util.graph import graph_feature

def test_neural_network():
  # graph_feature("neural_network/tol", range(1, 10000, 1000), 3000, lambda n: MLPClassifier(tol=1/n))
  # graph_feature("neural_network/learning_rate_init", range(1, 1500, 100), 10000, lambda n: MLPClassifier(learning_rate_init=(n/1000)))
  # graph_feature("neural_network/power_t", range(1, 1000, 100), 10000, lambda n: MLPClassifier(power_t=n/2))
  # graph_feature("neural_network/power_t", (100, 10000), [
  #   { "name": "Leaf Size 1", "learner": MLPClassifier(solver='sgd', learning_rate='adaptive', power_t=0.5) },
  #   # { "name": "Leaf Size 2", "learner": DecisionTreeClassifier(max_depth=10) },
  #   { "name": "Leaf Size 5", "learner": MLPClassifier(solver='sgd', learning_rate='adaptive', power_t=1) },
  # ])
  graph_feature("neural_network/max_iter", (100, 2000), [
    { "name": "Leaf Size 1", "learner": MLPClassifier(solver='sgd', learning_rate='adaptive', max_iter=200) },
    # { "name": "Leaf Size 2", "learner": DecisionTreeClassifier(max_depth=10) },
    { "name": "Leaf Size 5", "learner": MLPClassifier(solver='sgd', learning_rate='adaptive', max_iter=10000) },
  ])
#endfor
