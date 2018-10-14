import matplotlib.pyplot as plt
import os

out = os.popen('cat output/iterations.log | grep iterations').read().split('\n')
it = []
for i in range(0, len(out) - 1):
  it.append(float(out[i].split(' ')[-1]))
#endfor
print('iterations', it)

out = os.popen('cat output/iterations.log | grep "Train Error"').read().split('\n')
train = []
for i in range(0, len(out) - 1):
  train.append(float(out[i].split(' ')[-2][:-1]))
#endfor
print('train', train)

out = os.popen('cat output/iterations.log | grep "Test Error"').read().split('\n')
test = []
for i in range(0, len(out) - 1):
  test.append(float(out[i].split(' ')[-2][:-1]))
#endfor
print('test', test)

out = os.popen('cat output/iterations.log | grep "Time Elapsed"').read().split('\n')
time = []
for i in range(0, len(out) - 1):
  time.append(float(out[i].split(' ')[-3]))
#endfor
print('time', time)

n = int(len(it) / 4)
print()
print(n, 'points')

x = lambda: {'it': [], 'train': [], 'test': [], 'time': []}

backprop = x()
rhc = x()
sa = x()
ga = x()

y = [backprop, rhc, sa, ga]

for i in range(n):
  for j in range(4):
    y[j]['it'].append(it[4*i + j])
    y[j]['train'].append(train[4*i + j])
    y[j]['test'].append(test[4*i + j])
    y[j]['time'].append(time[4*i + j])
  #endfor
#endfor

titles = ['Backprop', 'Random Hill Climbing', 'Simulated Annealing', 'Genetic Algorithms']

for i in range(4):
  fig = plt.figure()
  plt.plot(y[i]['it'], y[i]['train'], ".--", label="Train")
  plt.plot(y[i]['it'], y[i]['test'], ".-", label="Test")
  plt.title(titles[i])
  plt.xlabel("Instances")
  plt.ylabel("Error")
  plt.ylim(0, 50)
  plt.legend()
  fig.savefig('graphs/{}'.format(titles[i]))
#endfor
