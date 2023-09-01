import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

parent_dir = '/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/Defect/1st-1st day/' \
             'Y 754.970 sec to 755.080 sec-1-1st test-20mph-1st day.csv'
a = pd.read_csv(parent_dir)
signal = a.iloc[:, 1]
output = stats.zscore(signal)

plt.figure(1)
plt.subplot(211)
plt.plot(output)
plt.subplot(212)
plt.plot(signal)
plt.show()
