import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fftpack import fft


def signal_processing_ten_seconds(output_path_sensor3, num_files):
    """cancel data shift and read split file"""
    figure_path = 'D:/Python/HTL/1st day/1st test-20mph-sensor3/two seconds data'
    os.makedirs(figure_path, exist_ok=True)
    iterations = num_files / 2
    sampling_rate = 1000000
    filter_frequency = 60000
    freq_to_pass = filter_frequency / sampling_rate * 2
    s = output_path_sensor3[60]
    for i in range(0, int(iterations)):
        ten_sec_signal = []
        ten_sec_signal_filtered = []
        for j in range(0, 2):
            duration = 2
            k = i * duration + j
            split_file = pd.read_csv(os.path.join(output_path_sensor3, str(k) + '-3' + '.csv'))
            acoustic_data = list(split_file.iloc[:, 1])
            # x = data[':,1']
            calibration = np.mean(acoustic_data)
            acoustic_data -= calibration
            ten_sec_signal.append(acoustic_data)
            """Band-pass filter"""
            b1, a1 = signal.butter(4, freq_to_pass, 'highpass')  # setup filter parameters
            acoustic_data_filtered = signal.filtfilt(b1, a1, acoustic_data)  # data is signal to be filtered
            ten_sec_signal_filtered.append(acoustic_data_filtered)
        print(i, i + duration)
        str1 = 'Signal ' + str((int(s) - 1) * 200 + i * duration) + ' to ' + str((int(s) - 1) * 200 + i * duration + duration) + ' sec'
        time_sequence = np.arange(i * duration, (i + 1) * duration, 1 / sampling_rate)
        ten_sec_signal = np.array(ten_sec_signal)
        ten_sec_signal_filtered = np.array(ten_sec_signal_filtered)
        ten_sec_signal = ten_sec_signal.flatten()
        ten_sec_signal_filtered = ten_sec_signal_filtered.flatten()

        """FFT calculation"""
        """Split each file into time units, and implement high-pass filter"""
        """plot filtered signal at time domain"""
        plt.figure(1)
        plt.subplot(211)
        plt.plot(time_sequence, ten_sec_signal)
        plt.title(str1)
        plt.ylabel('Amplitude(V)')
        # plt.ylim(-10, 10)

        plt.subplot(212)
        plt.plot(time_sequence, ten_sec_signal_filtered)
        plt.ylabel('Amplitude(V)')
        # plt.ylim(-10, 10)

        plt.savefig(os.path.join(figure_path, str1 + '.png'))
        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()  # Close a figure window
