import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt
from scipy import signal
from scipy.fftpack import fft
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MinMaxScaler

sampling_rate = 1000000
file_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/100 kHz/Defect/copy/4th-1st day/'
# '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/noise ' \
# 'source-duplicated/'


figure_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/100 kHz/Defect/copy/4th-1st day/plot/'
# '/media/lj/MachineLearning/AE recognition/Data/HTL data/11th/defect/plot1/'
#
os.makedirs(figure_path, exist_ok=True)
dir_list = os.listdir(file_path)
for filename in dir_list:
    if filename == 'plot':
        continue
    split_file = pd.read_csv(os.path.join(file_path, filename))
    # acoustic_data = list(split_file.iloc[:, 1])
    acoustic_data = split_file.iloc[:, 1].to_numpy()
    file_length = len(acoustic_data)
    """Band-pass filter"""
    # b, a = signal.butter(3, 0.06, 'highpass')  # setup filter parameters
    # acoustic_data_filtered = signal.filtfilt(b, a, acoustic_data)  # data is signal to be filtered
    # xpt_start = filename.split(' ')[0]
    # xpt_start = filename.split(' ')[1]
    # xpt_end = filename.split(' ')[3]
    # xpt_end = filename.split('@')[1].split('.')[0]
    pt_step = 1 / sampling_rate
    # fig_title = 'Defect ' + str(xpt_start) + '-' + str(xpt_end)
    fig_title = filename.split('.csv')[0]
    # str1 = 'Signal ' + str('%.2f' % xpt_start) + ' to ' + str(xpt_end) + ' sec'

    # time_sequence = np.arange(float(xpt_start), float(xpt_end), pt_step)
    time_sequence = np.arange(0, file_length / sampling_rate, pt_step)
    # defect_start = 0.05

    fig, ax = plt.subplots(figsize=(6, 4))
    # fig = plt.figure(figsize=(6, 3))
    plt.figure(1)
    # widths = np.arange(1, 31)
    # cwtmatr = signal.cwt(acoustic_data, signal.ricker, widths)
    # plt.imshow(cwtmatr, extent=[0, 0.11, 1, 31], cmap='jet', aspect='auto',
    #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    _, _, sgram = signal.spectrogram(acoustic_data, sampling_rate, nperseg=256, scaling='spectrum')
    logspec_total = librosa.amplitude_to_db(sgram)
    # librosa.display.specshow(logspec_total, sr=sampling_rate, x_axis='time', y_axis='linear', cmap='jet')
    plt.imshow(logspec_total, extent=[0, 0.11, 0, 500], cmap='jet', aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    fig.tight_layout()
    plt.savefig(os.path.join(figure_path, fig_title + '-spectrum.png'))
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure
    plt.close()  # Close a figure window

    plt.figure(2)
    plt.subplot(211)
    plt.plot(time_sequence[0: file_length], list(acoustic_data), zorder=1)
    plt.ylim(-1, 1)
    plt.title(fig_title)
    plt.xlabel('Time(sec)')
    plt.ylabel('Amplitude(V)')
    # ax.add_patch(Rectangle((defect_start, -9.5), 0.01, 19, fc='none', ec='r', lw=2, zorder=2))

    """calculate FFT"""
    freq_spec_1 = abs(fft(list(acoustic_data)))
    # freq_spec_2 = abs(fft(list(noise_data)))
    freq_spec_1 = 20 * np.log10(freq_spec_1)
    # freq_spec_2 = 20 * np.log10(freq_spec_2)
    num_datapts = int(np.fix(len(freq_spec_1) / 2))
    freq_spec_1 = freq_spec_1[0: num_datapts]
    # freq_spec_2 = freq_spec_2[0: num_datapts]
    fft_pts = np.arange(0, num_datapts)
    freq1 = fft_pts * (1000000 / (2 * num_datapts))

    """plot signal at frequency domain"""
    plt.subplot(212)
    xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
    xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
    plt.stem(freq1, freq_spec_1, markerfmt='None', basefmt='None')
    plt.xticks(xtick, xticklabels)
    plt.ylim(0, 50)
    plt.xlabel('Frequency(kHz)')
    plt.ylabel('Magnitude(dB)')
    plt.title('Frequency Spectrum')

    fig.tight_layout()
    plt.savefig(os.path.join(figure_path, fig_title + '.png'))
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure
    plt.close()  # Close a figure window

    """Wavelet packet analysis"""
    # dec_lv = 8
    # total_scales = 2 ** dec_lv
    # data_matrix = []
    # data_dec = pywt.WaveletPacket(data=acoustic_data, wavelet='db1', mode='symmetric', maxlevel=dec_lv)
    # for j in [node.path for node in data_dec.get_level(dec_lv, 'freq')]:
    #     data_array = data_dec[j].data
    #     data_matrix = np.append(data_matrix, data_array)
    # data_matrix = np.reshape(data_matrix, (total_scales, -1))
    # xlim = data_matrix.shape[1]
    # x_step = xlim / 5
    #
    # plt.figure(2)
    # plt.pcolormesh(np.abs(data_matrix), cmap='jet')
    # plt.colorbar()
    # xtick = np.arange(0, xlim, x_step)
    # xticklabels = np.arange(0, 0.05, 0.01)
    # xticklabels = np.around(xticklabels, decimals=1)
    # ytick = [0, 51.2, 102.4, 153.6, 204.8, 256]
    # yticklabels = ['0', '100', '200', '300', '400', '500']
    # plt.xticks(xtick, xticklabels)
    # plt.yticks(ytick, yticklabels)
    # plt.ylabel("freq(kHz)")
    # plt.xlabel("time(s)")
    # plt.subplots_adjust(hspace=0.4)
    # plt.savefig(os.path.join(figure_path, fig_title + 'wavelet.png'))
    # plt.cla()  # Clear axis
    # plt.clf()  # Clear figure
    # plt.close()  # Close a figure window
