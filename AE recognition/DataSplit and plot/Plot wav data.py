import os
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt
from scipy.fftpack import fft
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MinMaxScaler
from scipy.io import wavfile
from scipy import signal
import wave

# sampling_rate = 1000000
count = 5
file_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/test tdms/'
figure_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/plb-museum/plb/'
data, sampling_rate = librosa.core.load(os.path.join(file_path, 'plb-air.wav'), sr=384000)
audio_output = data
audio_length = int(data.shape[0] / sampling_rate)
os.makedirs(figure_path, exist_ok=True)
dir_list = os.listdir(file_path)
for i in range(0, audio_length):
    split_file = audio_output[i * sampling_rate: (i + 1) * sampling_rate]
    acoustic_data = list(split_file)
    for j in range(0, count):
        split_signal = acoustic_data[
                       int(np.fix(j / count * sampling_rate)): int(np.fix((j + 1) / count * sampling_rate))]
        start_time = str(i + j / 5)
        end_time = str(i + (j + 1) / 5)
        pt_step = 1 / sampling_rate
        fig_title = 'Defect ' + start_time + '-' + end_time
        # fig_title = 'Signal ' + str(xpt_start) + ' to ' + str(xpt_end) + ' sec'
        # str1 = 'Signal ' + str('%.2f' % xpt_start) + ' to ' + str(xpt_end) + ' sec'

        # time_sequence = np.arange(float(xpt_start), float(xpt_end), pt_step)
        time_sequence = np.arange(0, 0.2, pt_step)
        # defect_start = 0.05

        b, a = signal.butter(5, 0.16, 'highpass')  # setup filter parameters
        acoustic_data_filtered = signal.filtfilt(b, a, split_signal)  # data is signal to be filtered

        fig, ax = plt.subplots(figsize=(6, 4))
        # fig = plt.figure(figsize=(6, 3))
        plt.figure(1)
        plt.subplot(211)
        plt.plot(time_sequence[0: 76800], acoustic_data_filtered, zorder=1)
        # plt.ylim(-1, 1)
        plt.title(fig_title)
        plt.xlabel('Time(sec)')
        plt.ylabel('Amplitude(V)')
        # ax.add_patch(Rectangle((defect_start, -9.5), 0.01, 19, fc='none', ec='r', lw=2, zorder=2))

        """calculate FFT"""
        freq_spec = abs(fft(list(acoustic_data_filtered)))
        # freq_spec = 20 * np.log10(freq_spec)
        num_datapts = int(np.fix(len(freq_spec) / 2))
        freq_spec = freq_spec[0: num_datapts]
        fft_pts = np.arange(0, num_datapts)
        freq1 = fft_pts * (sampling_rate / (2 * num_datapts))

        """plot signal at frequency domain"""
        plt.subplot(212)
        xtick = [0, 50000, 100000, 150000, 200000]
        xticklabels = ['0', '50', '100', '150', '200']
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

