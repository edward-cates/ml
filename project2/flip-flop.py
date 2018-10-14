import matplotlib.pyplot as plt
import os

out = os.popen('cat output/part2/flipflop.log | grep iterations').read().split('\n')

title = 'Flip Flop'

titles = ['Random Hill Climbing', 'Simulated Annealing', 'Genetic Algorithms', 'MIMIC']


for i in range(4):
  xs = []
  ys = []
  for j in range(10):
    ix = 10*i + j
    words = out[ix].split(' ')

    xs.append(int(words[0]))
    ys.append(float(words[-1]))
  #endfor
  label = '{}: {}'.format(title, titles[i])

  fig = plt.figure()
  plt.plot(xs, ys)
  plt.title(label)
  plt.xlabel('Iterations')
  plt.ylabel('Value')
  plt.ylim(60, 80)
  fig.savefig('graphs/part2/{}'.format(label))
#endfor
