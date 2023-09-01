import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fftpack import fft


"""cancel data shift and read split file"""
# raw_data_path = 'D:/Python/HTL/1st day/2nd test whole data/raw/'
filtered_data_path = 'D:/Python/RDTF/5 mph//3rd test/filtered/'
output_path_sensor = 'D:/Python/RDTF/5 mph//3rd test/'
# os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(filtered_data_path, exist_ok=True)
sampling_rate = 1000000
filter_frequency = 60000
freq_to_pass = filter_frequency / sampling_rate * 2
# two_hundred_sec_signal = []
two_hundred_sec_signal_filtered = []
for j in range(0, 192):
    split_file = pd.read_csv(os.path.join(output_path_sensor, str(j) + '-2' + '.csv'))
    acoustic_data = list(split_file.iloc[:, 1])
    # x = data[':,1']
    calibration = np.mean(acoustic_data)
    acoustic_data -= calibration
    # two_hundred_sec_signal.append(acoustic_data)
    """Band-pass filter"""
    b1, a1 = signal.butter(12, freq_to_pass, 'highpass')  # setup filter parameters
    acoustic_data_filtered = signal.filtfilt(b1, a1, acoustic_data)  # data is signal to be filtered
    two_hundred_sec_signal_filtered.append(acoustic_data_filtered)

# two_hundred_sec_signal = np.array(two_hundred_sec_signal)
two_hundred_sec_signal_filtered = np.array(two_hundred_sec_signal_filtered)
# two_hundred_sec_signal = two_hundred_sec_signal.flatten()
two_hundred_sec_signal_filtered = two_hundred_sec_signal_filtered.flatten()

# file1 = open("D:/Python/HTL/1st day/2nd test whole data/raw/data-4-sensor3.txt", "w")
# for i in range(0, len(two_hundred_sec_signal)):
#     raw = two_hundred_sec_signal[i]
#     content = str(raw)
#     file1.write(content + "\n")
#     print(i)
# file1.close()

file2 = open("D:/Python/HTL/1st day/2nd test whole data/filtered/data-4-sensor3.txt", "w")
for i in range(0, len(two_hundred_sec_signal_filtered)):
    filtered = two_hundred_sec_signal_filtered[i]
    content = str(filtered)
    file2.write(content + "\n")
    print(i)
file2.close()
