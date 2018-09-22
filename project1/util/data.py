import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# data = pd.read_csv('absent.csv')
data = pd.read_csv('cc-defaults.csv')

x = data.ix[:, :-1].values
y = data.ix[:, -1].values

# DOWNSAMPLING could be done here
yes = data[y == 1]
no = data[y == 0]

print('yes', len(yes))
print('no', len(no))

if len(yes) > len(no):
  yes = resample(yes, replace=False, n_samples=len(no))
else:
  no = resample(no, replace=False, n_samples=len(yes))
#endif

data = pd.concat([yes, no])

x = data.ix[:, :-1].values
y = data.ix[:, -1].values
# DOWNSAMPLING complete

def get_data(rows):
  # should sample while maintaining proportions
  fold_x, _, fold_y, _ = train_test_split(x, y, train_size=(rows / data.shape[0]), test_size=0.1)

  # UPSAMPLING could be done here

  return train_test_split(fold_x, fold_y, train_size=0.7, test_size=0.3)
#enddef
