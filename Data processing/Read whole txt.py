import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fftpack import fft


"""cancel data shift and read split file"""
figure_path = 'D:/Python/HTL/1st day/2nd test whole data/filtered/'
output_path_sensor = 'D:/Python/HTL/1st day/2nd test whole data/filtered/'
os.makedirs(figure_path, exist_ok=True)
sampling_rate = 1000000
filter_frequency = 60000
freq_to_pass = filter_frequency / sampling_rate * 2
split_file = []
for j in range(1, 6):
    with open(os.path.join(output_path_sensor, 'data-' + str(j) + '-sensor2' + '.txt')) as f:
        split_file.append(f.readlines())

str1 = 'Signal 0 to 600 sec'
time_sequence = np.arange(0, 600, 1 / sampling_rate)

"""FFT calculation"""
"""Split each file into time units, and implement high-pass filter"""
"""plot filtered signal at time domain"""
plt.figure(1)
fig = plt.figure(figsize=(12, 6))
xtick = [0, 100, 200, 300, 400, 500, 600]
plt.plot(time_sequence, split_file, 'k')
plt.xlabel('Second')
plt.ylabel('Amplitude(V)')
# plt.ylim(-10, 10)
fig.tight_layout()

plt.savefig(os.path.join(figure_path, str1 + '.png'))
plt.cla()  # Clear axis
plt.clf()  # Clear figure
plt.close()  # Close a figure window
