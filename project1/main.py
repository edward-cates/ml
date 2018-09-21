import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

from graph import plot_learning_curve

matplotlib.rcParams.update({'font.size': 6})
fig = plt.figure()

# data = pd.read_csv('absent.csv')
data = pd.read_csv('cc-defaults.csv')
# up-sample (https://elitedatascience.com/imbalanced-classes)
# y = data.ix[:, -1]
# df_maj = data[y == 0]
# df_min = data[y == 1]
# df_min_up = resample(df_min, replace = True, n_samples = df_maj.shape[0])
# data = pd.concat([df_maj, df_min_up])

x = data.ix[:, :-1].values
y = data.ix[:, -1].values

algos = [
  {
    "label": "Dummy",
    "learner": DummyClassifier(),
  },
  {
    "label": "Boosting",
    "learner": AdaBoostClassifier(),
  },
  {
    "label": "k Nearest Neighbors",
    "learner": KNeighborsClassifier(n_neighbors=5),
  },
  {
    "label": "Decision Tree",
    "learner": DecisionTreeClassifier(),
  },
  {
    "label": "Neural Network",
    "learner": MLPClassifier(tol=1e-6,learning_rate_init=0.001),#hidden_layer_sizes=(1000,500,100,50)
  },
  {
    "label": "SVM",
    # "learner": SVC(kernel='linear', tol=1),
    "learner": LinearSVC(dual=False, max_iter=9999999),
  },
]

rows = []
for i in range(0, 10, 1):
  rows.append(100 * (i + 1))

count = 0
max = len(algos)

for algo in algos:
  print(algo["label"])

  plot_learning_curve(algo["learner"], algo["label"], x, y, n_jobs=8, train_sizes=rows)

  n_rows = []
  train_errors = []
  test_errors = []
  cross_val_errors = []

  for total_rows in rows:
    print(total_rows)
    base_ratio = total_rows / x.shape[0]

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.3 * base_ratio, train_size=.7 * base_ratio)

    train_data = np.append(train_x, np.zeros((train_x.shape[0], 1)), axis=1)
    train_data[:, -1] = train_y

    df_maj = train_data[train_y == 0]
    df_min = train_data[train_y == 1]
    df_min_up = resample(df_min, replace = True, n_samples = df_maj.shape[0])
    train_data = np.append(df_maj, df_min_up, axis=0)

    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]

    algo["learner"].fit(train_x, train_y)
    train_score = algo["learner"].score(train_x, train_y)
    test_score = algo["learner"].score(test_x, test_y)

    n_rows.append(total_rows)
    train_errors.append(1 - train_score)
    test_errors.append(1 - test_score)

    tmp_x = np.append(train_x, test_x, axis=0)
    tmp_y = np.append(train_y, test_y, axis=0)
    cv_score = cross_val_score(algo["learner"], tmp_x, tmp_y, cv=10).mean()
    cross_val_errors.append(1 - cv_score)
  #endfor

  count += 1
  plt = fig.add_subplot(math.ceil(max / 2), 2, count)
  plt.set_title(algo["label"])
  plt.set_xlabel("Instances")
  plt.set_ylabel("Error")
  plt.plot(n_rows, train_errors, "o-", label="Train")
  # plt.plot(n_rows, test_errors, label="Test")
  plt.plot(n_rows, cross_val_errors, "o-", label="Cross Val")
  plt.legend()
#endfor

fig.savefig("graphs.pdf")
