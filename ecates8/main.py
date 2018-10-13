import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample

from util.graph import plot_learning_curve

from learners.decision_tree import test_decision_tree
from learners.k_nearest_neighbors import test_knn
from learners.neural_network import test_neural_network
from learners.svm import test_svm
from learners.boosting import test_boosting

matplotlib.rcParams.update({'font.size': 6})

print('=============')
print('decision tree')
print('-------------')
test_decision_tree()

print('===')
print('kNN')
print('---')
test_knn()

print('========')
print('boosting')
print('--------')
test_boosting()

print('===')
print('SVM')
print('---')
test_svm()

print('==============')
print('neural network')
print('--------------')
test_neural_network()
