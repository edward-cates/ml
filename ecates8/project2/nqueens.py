import matplotlib.pyplot as plt
import os

out = os.popen('cat output/part2/nqueens.log | grep iterations').read().split('\n')

title = 'N Queens'

titles = ['Random Hill Climbing', 'Simulated Annealing', 'Genetic Algorithms', 'MIMIC']


for i in range(4):
  xs = []
  ys = []
  for j in range(10):
    ix = 10*i + j
    words = out[ix].split(' ')

    xs.append(int(words[0]))
    ys.append(float(words[-2]))
  #endfor
  label = '{}: {}'.format(title, titles[i])

  fig = plt.figure()
  plt.plot(xs, ys)
  plt.title(label)
  plt.xlabel('Iterations')
  plt.ylabel('Value')
  plt.ylim(150, 200)
  fig.savefig('graphs/part2/{}'.format(label))
#endfor
