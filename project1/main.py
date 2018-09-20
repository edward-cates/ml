import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.rcParams.update({'font.size': 6})
fig = plt.figure()

data = pd.read_csv('cc-defaults.csv')
x = data.ix[:, :-1]
y = data.ix[:, -1]

algos = [
  # {
  #   "label": "Dummy",
  #   "learner": DummyClassifier(),
  # },
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
    "learner": DecisionTreeClassifier(max_leaf_nodes=None),
  },
  {
    "label": "Neural Network",
    "learner": MLPClassifier(),
  },
  # {
  #   "label": "SVM",
  #   "learner": SVC(kernel='linear', tol=1),
  # },
]

rows = []
for i in range(0, 40, 1):
  rows.append(250 * (1 + i))

count = 0
max = len(algos)

for algo in algos:
  print(algo["label"])

  n_rows = []
  errors = []

  for total_rows in rows:
    train_rows = math.floor(.7 * total_rows)
    test_rows = math.floor(.3 * total_rows)

    train_x = x[:train_rows]
    train_y = y[:train_rows]
    test_x = x[-test_rows:]
    test_y = y[-test_rows:]

    algo["learner"].fit(train_x, train_y)
    score = algo["learner"].score(test_x, test_y)

    n_rows.append(total_rows)
    errors.append(1 - score)
  #endfor

  count += 1
  plt = fig.add_subplot(math.ceil(max / 2), 2, count)
  plt.set_title(algo["label"])
  plt.plot(n_rows, errors)
#endfor

fig.savefig("graphs.png")
