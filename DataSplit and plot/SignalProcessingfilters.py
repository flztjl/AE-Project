import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fftpack import fft
import pywt


def signal_processing_one_second(output_path_sensor2, num_files):
    """cancel data shift and read split file"""
    figure_path = os.path.join(output_path_sensor2, 'one second data')
    os.makedirs(figure_path, exist_ok=True)
    for i in range(0, num_files):
        split_file = pd.read_csv(os.path.join(output_path_sensor2, str(i) + '-2' + '.csv'))
        acoustic_data = list(split_file.iloc[:, 1])
        # x = data[':,1']
        calibration = np.mean(acoustic_data)
        acoustic_data -= calibration
        sampling_rate = 1000000
        time_sequence = np.arange(0, 1, 1 / sampling_rate)
        str1 = 'Signal ' + str('%.2f' % i) + ' to ' + str(i + 1) + ' sec'
        print(i, i + 1)

        """Band-pass filter"""
        b1, a1 = signal.butter(18, 0.16, 'highpass')  # setup filter parameters
        acoustic_data_filtered1 = signal.filtfilt(b1, a1, acoustic_data)  # data is signal to be filtered

        """Band-pass filter"""
        b2, a2 = signal.butter(10, [0.16, 0.3], 'bandpass')  # setup filter parameters
        acoustic_data_filtered2 = signal.filtfilt(b2, a2, acoustic_data)  # data is signal to be filtered

        """Band-pass filter"""
        b3, a3 = signal.butter(10, [0.3, 0.9], 'bandpass')  # setup filter parameters
        acoustic_data_filtered3 = signal.filtfilt(b3, a3, acoustic_data)  # data is signal to be filtered

        """Wavelet packet analysis"""
        # dec_lv = 8
        # total_scales = 2 ** dec_lv
        # data_matrix = []
        # data_dec = pywt.WaveletPacket(data=acoustic_data, wavelet='db1', mode='symmetric', maxlevel=dec_lv)
        # for j in [node.path for node in data_dec.get_level(dec_lv, 'freq')]:
        #     data_array = data_dec[j].data
        #     data_matrix = np.append(data_matrix, data_array)
        # data_matrix = np.reshape(data_matrix, (total_scales, -1))
        # # data_matrix[0:40] = 0
        # # data_matrix[153:256] = 0
        # xlim = data_matrix.shape[1]
        # x_step = xlim / 10

        # plt.figure(1)
        # plt.pcolor(abs(data_matrix), cmap='Blues')
        # plt.colorbar()
        # xtick = np.arange(0, xlim, x_step)
        # xticklabels = np.arange(float(i), float(i + 1), 0.1)
        # xticklabels = np.around(xticklabels, decimals=1)
        # plt.xticks(xtick, xticklabels, rotation=-90)
        # plt.ylabel("freq(kHz)")
        # plt.xlabel("time(s)")
        # plt.subplots_adjust(hspace=0.4)
        # plt.savefig(os.path.join(figure_path, str(i) + '.png'))
        # plt.cla()  # Clear axis
        # plt.clf()  # Clear figure
        # plt.close()  # Close a figure window

        """FFT calculation"""
        """Split each file into time units, and implement high-pass filter"""
        """plot filtered signal at time domain"""
        plt.figure(1)
        plt.subplot(411)
        plt.plot(time_sequence, acoustic_data)
        plt.title(str1)
        plt.ylabel('Amplitude(V)')

        plt.subplot(412)
        plt.plot(time_sequence, acoustic_data_filtered1)
        plt.ylabel('Amplitude(V)')

        plt.subplot(413)
        plt.plot(time_sequence, acoustic_data_filtered2)
        plt.ylabel('Amplitude(V)')

        plt.subplot(414)
        plt.plot(time_sequence, acoustic_data_filtered3)
        plt.xlabel('Time(sec)')
        plt.ylabel('Amplitude(V)')

        plt.savefig(os.path.join(figure_path, str1 + '.png'))
        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()  # Close a figure window
