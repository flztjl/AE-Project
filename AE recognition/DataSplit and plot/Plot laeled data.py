import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt
from scipy.fftpack import fft
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MinMaxScaler

sampling_rate = 1000000
file_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/15th/Defect/5th-1st day/'
figure_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/15th/Defect/5th-1st day/plot/'
os.makedirs(figure_path, exist_ok=True)
dir_list = os.listdir(file_path)
for filename in dir_list:
    if filename == 'plot':
        continue
    data_type = filename.split(' ')[0]
    if data_type == 'Y':
        split_file = pd.read_csv(os.path.join(file_path, filename))
        acoustic_data = list(split_file.iloc[:, 1])
        xpt_start = filename.split(' ')[1]
        xpt_end = filename.split('@')[1].split('.')[0]
        pt_step = 1 / sampling_rate
        fig_title = 'Defect ' + str(xpt_start) + '-' + str(xpt_end)
        # fig_title = 'Signal ' + str(xpt_start) + ' to ' + str(xpt_end) + ' sec'
        # str1 = 'Signal ' + str('%.2f' % xpt_start) + ' to ' + str(xpt_end) + ' sec'

        # time_sequence = np.arange(float(xpt_start), float(xpt_end), pt_step)
        time_sequence = np.arange(0, 0.11, pt_step)
        defect_start = 0.05

        fig, ax = plt.subplots(figsize=(6, 4))
        # fig = plt.figure(figsize=(6, 3))
        plt.figure(1)
        plt.subplot(211)
        plt.plot(time_sequence[0: 110000], list(acoustic_data), zorder=1)
        plt.ylim(-1, 1)
        plt.title(fig_title)
        plt.xlabel('Time(sec)')
        plt.ylabel('Amplitude(V)')
        ax.add_patch(Rectangle((defect_start, -9.5), 0.01, 19, fc='none', ec='r', lw=2, zorder=2))

        """calculate FFT"""
        freq_spec = abs(fft(list(acoustic_data)))
        # freq_spec = 20 * np.log10(freq_spec)
        num_datapts = int(np.fix(len(freq_spec) / 2))
        freq_spec = freq_spec[0: num_datapts]
        fft_pts = np.arange(0, num_datapts)
        freq1 = fft_pts * (sampling_rate / (2 * num_datapts))

        """plot signal at frequency domain"""
        plt.subplot(212)
        xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
        xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
        plt.stem(freq1, freq_spec, markerfmt='None', basefmt='None')
        plt.xticks(xtick, xticklabels)
        # plt.ylim(0, 50)
        plt.xlabel('Frequency(kHz)')
        plt.ylabel('Magnitude(dB)')
        plt.title('Frequency Spectrum')

        fig.tight_layout()
        plt.savefig(os.path.join(figure_path, fig_title + '.png'))
        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()  # Close a figure window

        """Wavelet packet analysis"""

        # acoustic_data_section = acoustic_data[int(i * 0.02 * sampling_rate): int((i + 1) * 0.02 * sampling_rate)]
        # start_pt = xpt_start + "{:.2f}".format(i * 0.02)
        # end_pt = xpt_start + "{:.2f}".format((i + 1) * 0.02)
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
        # x_step = xlim / 11
        #
        # plt.figure(2)
        # plt.imshow(np.abs(data_matrix))
        # plt.colorbar()
        # xtick = np.arange(0, xlim, x_step)
        # xticklabels = np.arange(0, 0.11, 0.01)
        # xticklabels = np.around(xticklabels, decimals=1)
        # plt.xticks(xtick, xticklabels, rotation=-90)
        # plt.ylabel("freq(kHz)")
        # plt.xlabel("time(s)")
        # plt.subplots_adjust(hspace=0.4)
        # plt.savefig(os.path.join(figure_path, fig_title + 'wavelet.png'))
        # plt.cla()  # Clear axis
        # plt.clf()  # Clear figure
        # plt.close()  # Close a figure window

    else:
        continue
