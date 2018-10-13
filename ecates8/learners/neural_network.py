from sklearn.neural_network import MLPClassifier
from util.graph import graph_feature

def test_neural_network():
  graph_feature("neural_network/power_t", (100, 10000), [
    { "name": "Power T 0.25", "learner": MLPClassifier(solver='sgd', learning_rate='adaptive', power_t=0.25) },
    { "name": "Power T 2.0", "learner": MLPClassifier(solver='sgd', learning_rate='adaptive', power_t=2) },
  ])
  graph_feature("neural_network/max_iter", (100, 2000), [
    { "name": "200 Iterations", "learner": MLPClassifier(solver='sgd', activation='tanh', learning_rate='adaptive', max_iter=200) },
    { "name": "10000 Iterations", "learner": MLPClassifier(solver='sgd', activation='tanh', learning_rate='adaptive', max_iter=10000) },
  ])
  graph_feature("neural_network/hidden_layers", (100, 2000), [
    { "name": "(40,40,40) Hidden Layer", "learner": MLPClassifier(solver='sgd', learning_rate='adaptive', hidden_layer_sizes=(40, 40, 40)) },
    { "name": "(100,100) Hidden Layer", "learner": MLPClassifier(solver='sgd', learning_rate='adaptive', hidden_layer_sizes=(100, 100)) },
  ])
#endfor
