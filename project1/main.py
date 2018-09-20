import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

matplotlib.rcParams.update({'font.size': 6})
fig = plt.figure()

# data = pd.read_csv('absent.csv')
data = pd.read_csv('cc-defaults.csv')
# up-sample (https://elitedatascience.com/imbalanced-classes)
y = data.ix[:, -1]
df_maj = data[y == 0]
df_min = data[y == 1]
df_min_up = resample(df_min, replace = True, n_samples = df_maj.shape[0])
data = pd.concat([df_maj, df_min_up])

x = data.ix[:, :-1].values
y = data.ix[:, -1].values

algos = [
  # {
  #   "label": "Dummy",
  #   "learner": DummyClassifier(),
  # },
  # {
  #   "label": "Boosting",
  #   "learner": AdaBoostClassifier(),
  # },
  # {
  #   "label": "k Nearest Neighbors",
  #   "learner": KNeighborsClassifier(n_neighbors=5),
  # },
  {
    "label": "Decision Tree",
    "learner": DecisionTreeClassifier(),
  },
  # {
  #   "label": "Neural Network",
  #   "learner": MLPClassifier(),
  # },
  # {
  #   "label": "SVM",
  #   "learner": SVC(kernel='linear', tol=1),
  # },
]

rows = []
for i in range(0, 30, 1):
  rows.append(1000 * (1 + i))

count = 0
max = len(algos)

for algo in algos:
  print(algo["label"])

  n_rows = []
  train_errors = []
  test_errors = []
  cross_val_errors = []

  for total_rows in rows:
    # used to get a subset of the data of size total_rows
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(total_rows/x.shape[0]))
    for train_index, test_index in sss1.split(x, y):
      # tmp data subset of size total_rows
      tmp_x, tmp_y = x[test_index], y[test_index]

      sss2 = StratifiedShuffleSplit(n_splits=1, test_size=.3)
      for train_index2, test_index2 in sss2.split(tmp_x, tmp_y):
        train_x, train_y = tmp_x[train_index2], tmp_y[train_index2]
        test_x, test_y = tmp_x[test_index2], tmp_y[test_index2]

        algo["learner"].fit(train_x, train_y)
        train_score = algo["learner"].score(train_x, train_y)
        test_score = algo["learner"].score(test_x, test_y)

        n_rows.append(total_rows)
        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)
      #endfor

      cv_score = cross_val_score(algo["learner"], tmp_x, tmp_y, cv=10).mean()
      cross_val_errors.append(1 - cv_score)
    #endfor
  #endfor

  count += 1
  plt = fig.add_subplot(math.ceil(max / 2), 1, count)
  plt.set_title(algo["label"])
  plt.set_xlabel("Instances")
  plt.set_ylabel("Error")
  plt.plot(n_rows, train_errors, label="Train")
  plt.plot(n_rows, test_errors, label="Test")
  plt.plot(n_rows, cross_val_errors, label="Cross Val")
  plt.legend()
#endfor

fig.savefig("graphs.pdf")
