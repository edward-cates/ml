import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 22})

output = [
  [2, 63.55124497991967],
  [4, 63.911646586345384],
  [6, 66.01317269076306],
  [8, 65.77277108433735],
  [10, 66.03325301204819],
  [12, 65.85333333333334],
  [14, 66.1532530120482],
  [2, 55.8378313253012],
  [4, 52.980000000000004],
  [6, 53.638554216867476],
  [8, 58.12819277108433],
  [10, 60.804497991967864],
  [12, 57.60586345381527],
  [14, 53.48056224899598],
  [2, 58.90674698795182],
  [4, 85.15277108433736],
  [6, 87.13518072289158],
  [8, 90.43518072289156],
  [10, 93.07815261044178],
  [12, 92.47421686746989],
  [14, 92.89485943775102],
  [2, 50.0],
  [4, 50.0],
  [6, 50.0],
  [8, 50.0],
  [10, 50.0],
  [12, 50.0],
  [14, 50.0],
]

reducers = [
  { "label": "FeatCluster", "reducer": 0 },
  { "label": "ICA", "reducer": 0 },
  { "label": "PCA", "reducer": 0 },
  { "label": "Random", "reducer": 0 },
]

n = 7

for i in range(len(reducers)):
  start, end = i*n, (i+1)*n
  values = np.array(output[start:end])
  plt.plot(values[:, 0], values[:, 1], label=reducers[i]["label"])
#endfor

plt.ylim([40, 100])
plt.legend()
plt.show()
