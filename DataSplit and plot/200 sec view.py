import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fftpack import fft


"""cancel data shift and read split file"""
figure_path = 'D:/Python/HTL/1st day/2nd test whole data/filtered/'
output_path_sensor = 'D:/Python/HTL/1st day/1st test-20mph-sensor2/1st test-20mph-3/'
os.makedirs(figure_path, exist_ok=True)
sampling_rate = 1000000
filter_frequency = 60000
freq_to_pass = filter_frequency / sampling_rate * 2
ten_sec_signal = []
ten_sec_signal_filtered = []
for j in range(150, 200):
    split_file = pd.read_csv(os.path.join(output_path_sensor, str(j) + '-2' + '.csv'))
    acoustic_data = list(split_file.iloc[:, 1])
    # x = data[':,1']
    calibration = np.mean(acoustic_data)
    acoustic_data -= calibration
    ten_sec_signal.append(acoustic_data)
    """Band-pass filter"""
    b1, a1 = signal.butter(12, freq_to_pass, 'highpass')  # setup filter parameters
    acoustic_data_filtered = signal.filtfilt(b1, a1, acoustic_data)  # data is signal to be filtered
    ten_sec_signal_filtered.append(acoustic_data_filtered)

ten_sec_signal_filtered = np.array(ten_sec_signal_filtered)
ten_sec_signal_filtered = ten_sec_signal_filtered.flatten()

str1 = 'Signal 150 to 200 sec'
time_sequence = np.arange(150, 200, 1 / sampling_rate)


"""FFT calculation"""
"""Split each file into time units, and implement high-pass filter"""
"""plot filtered signal at time domain"""
plt.figure(1)
fig = plt.figure(figsize=(12, 6))
plt.plot(time_sequence, ten_sec_signal_filtered, 'k')
xtick = [150, 160, 170, 180, 190, 200]
xticklabels = ['550', '560', '570', '580', '590', '600']
plt.xticks(xtick, xticklabels)
plt.xlabel('Second')
plt.ylabel('Amplitude(V)')
# plt.ylim(-10, 10)
fig.tight_layout()

plt.savefig(os.path.join(figure_path, str1 + '.png'))
plt.cla()  # Clear axis
plt.clf()  # Clear figure
plt.close()  # Close a figure window
