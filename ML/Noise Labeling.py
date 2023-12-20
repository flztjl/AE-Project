import os
import math
import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from decimal import Decimal
from scipy.fftpack import fft
import matplotlib.pyplot as plt

output_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/noise/'
os.makedirs(output_path, exist_ok=True)
noise_data_path = '/media/lj/Data/Python/HTL/1st day/labeling-4th-test (copy).csv'
test_name = '4th-test-40mph'

noise_data = pd.read_csv(noise_data_path)
noise_time_list = list(noise_data.iloc[:, 4])
for i in range(0, 5):
    noise_time = noise_time_list[i]
    noise_time_point = int(noise_time % 200)
    integer = int(noise_time // 200 + 1)
    fold_number = '-' + str(integer)
    for j in range(1, 4):
        sensor_set = j
        input_path = '/media/lj/Data/Python/HTL/1st day/' + test_name + '-sensor' + str(sensor_set) + \
                     '/' + test_name + fold_number + \
                     '/' + str(noise_time_point) + '-' + str(sensor_set) + '.csv'
        split_file = pd.read_csv(input_path)
        acoustic_data = list(split_file.iloc[:, 1])

        """Scale"""
        acoustic_data_df = pd.DataFrame(acoustic_data)
        acoustic_data_df = acoustic_data_df.apply(lambda x: (x - min(x)) / (max(x) - min(x)), axis=0)
        acoustic_data_scaled = list(acoustic_data_df.iloc[:, 1])
        calibration = np.mean(acoustic_data_scaled)
        acoustic_data_scaled -= calibration
        timelength = len(acoustic_data_scaled)

        """Band-pass filter"""
        b, a = signal.butter(6, 0.2, 'highpass')  # setup filter parameters
        acoustic_data_filtered = signal.filtfilt(b, a, acoustic_data)  # data is signal to be filtered

        for k in range(0, 20):
            m = 0.05 * k
            x = 0.05 * (k + 1)
            # start_time = int(noise_time_point) + m
            # end_time = int(noise_time_point) + x
            data_subset = {'sensor': acoustic_data_filtered[int(m * 1000000): int(x * 1000000)]}
            dataframe = pd.DataFrame(data_subset)
            dataframe.to_csv(os.path.join(output_path, 'X ' + str(noise_time + m) + ' sec to '
                                          + str(noise_time + x) + ' sec-' + '-' + test_name +
                                          '-2nd day@' + str(sensor_set) + '.csv'))
