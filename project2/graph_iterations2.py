import matplotlib.pyplot as plt
import os

out = os.popen('cat output/part2/flipflop.log | grep iterations').read().split('\n')
it = []
for i in range(0, len(out) - 1):
  # it.append(float(out[i].split(' ')[-1]))
  print(out[i])
#endfor
print('iterations', it)

titles = ['Backprop', 'Random Hill Climbing', 'Simulated Annealing', 'Genetic Algorithms']

# for i in range(4):
#   fig = plt.figure()
#   plt.plot(y[i]['it'], y[i]['train'], ".--", label="Train")
#   plt.plot(y[i]['it'], y[i]['test'], ".-", label="Test")
#   plt.title(titles[i])
#   plt.xlabel("Instances")
#   plt.ylabel("Error")
#   plt.ylim(0, 50)
#   plt.legend()
#   fig.savefig('graphs/{}'.format(titles[i]))
#endfor
