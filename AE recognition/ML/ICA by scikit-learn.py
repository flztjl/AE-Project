import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from scipy import signal
from scipy.fftpack import fft
from sklearn.decomposition import FastICA

split_file_1 = pd.read_csv('D:/Google Drive/Colab Notebooks/AE recognition/Data/3/52.csv')
# split_file_2 = pd.read_csv('D:/Python/HTL/1st day/2nd test-20mph-sensor2/2nd test-20mph-3/101-2.csv')
# split_file_3 = pd.read_csv('D:/Python/HTL/1st day/2nd test-20mph-sensor3/2nd test-20mph-3/101-3.csv')

acoustic_data_1 = list(split_file_1.iloc[:, 1])
calibration_1 = np.mean(acoustic_data_1)
acoustic_data_1 -= calibration_1

acoustic_data_2 = list(split_file_1.iloc[:, 2])
calibration_2 = np.mean(acoustic_data_2)
acoustic_data_2 -= calibration_2

acoustic_data_3 = list(split_file_1.iloc[:, 3])
calibration_3 = np.mean(acoustic_data_3)
acoustic_data_3 -= calibration_3

figure_path = os.path.join('C:/Users/ssrs_/OneDrive/桌面/1/')
# 'D:/Python/HTL/1st day/Defects/2nd test/ICA/'
os.makedirs(figure_path, exist_ok=True)

sampling_rate = 1000000

i = Decimal('0.3')
j = Decimal('0.35')
k = math.ceil(sampling_rate * (j - i))
time_stamp = i

split_signal_1 = acoustic_data_1[int(i * sampling_rate): int(j * sampling_rate)]
split_signal_2 = acoustic_data_2[int(i * sampling_rate): int(j * sampling_rate)]
split_signal_3 = acoustic_data_3[int(i * sampling_rate): int(j * sampling_rate)]

"""Band-pass filter"""
b1, a1 = signal.butter(18, 0.08, 'highpass')  # setup filter parameters
acoustic_data_filtered1 = signal.filtfilt(b1, a1, split_signal_1)  # data is signal to be filtered

"""Band-pass filter"""
b2, a2 = signal.butter(18, 0.08, 'highpass')  # setup filter parameters
acoustic_data_filtered2 = signal.filtfilt(b2, a2, split_signal_2)  # data is signal to be filtered

"""Band-pass filter"""
b3, a3 = signal.butter(18, 0.08, 'highpass')  # setup filter parameters
acoustic_data_filtered3 = signal.filtfilt(b3, a3, split_signal_3)  # data is signal to be filtered

# hx = fftpack.hilbert(split_signal)
# hy = np.abs(hx)
# hy = np.sqrt(split_signal**2+hx**2)
xpt_start = i
xpt_end = j
pt_step = 1 / sampling_rate
str1 = 'Signal ' + str('%.2f' % time_stamp) + ' to ' + str(time_stamp + j - i) + ' sec'
str2 = 'Signal ' + str('%.2f' % time_stamp) + ' to ' + str(time_stamp + j - i) + ' sec-1'
time_sequence = np.arange(float(xpt_start), float(xpt_end), pt_step)
time_line = time_sequence[0: k]

refined_signal = np.c_[acoustic_data_filtered1, acoustic_data_filtered2, acoustic_data_filtered3]

ica = FastICA(n_components=3, max_iter=4000, tol=0.001)
S_ = ica.fit_transform(refined_signal)

"""calculate FFT"""
freq_spec_0 = abs(fft(S_[:, 0].T))
freq_spec_0 = 20 * np.log10(freq_spec_0)
num_datapts = int(np.fix(len(freq_spec_0) / 2))
freq_spec_0 = freq_spec_0[0: num_datapts]
fft_pts = np.arange(0, num_datapts)
freq_0 = fft_pts * (sampling_rate / (2 * num_datapts))

"""calculate FFT"""
freq_spec_1 = abs(fft(S_[:, 1].T))
freq_spec_1 = 20 * np.log10(freq_spec_1)
num_datapts = int(np.fix(len(freq_spec_1) / 2))
freq_spec_1 = freq_spec_1[0: num_datapts]
fft_pts = np.arange(0, num_datapts)
freq_1 = fft_pts * (sampling_rate / (2 * num_datapts))

"""calculate FFT"""
freq_spec_2 = abs(fft(S_[:, 2].T))
freq_spec_2 = 20 * np.log10(freq_spec_2)
num_datapts = int(np.fix(len(freq_spec_2) / 2))
freq_spec_2 = freq_spec_0[0: num_datapts]
fft_pts = np.arange(0, num_datapts)
freq_2 = fft_pts * (sampling_rate / (2 * num_datapts))

fig = plt.figure(figsize=(12, 6))
plt.figure(1)
plt.subplot(421)
plt.plot(time_line, refined_signal[:, 1])
plt.title(str1)
plt.ylabel('Amplitude(V)')

plt.subplot(423)
plt.plot(time_line, S_[:, 0].T)
plt.ylabel('Amplitude(V)')

"""plot signal at frequency domain"""
plt.subplot(424)
xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
plt.stem(freq_0, freq_spec_0, markerfmt='None', basefmt='None')
plt.xticks(xtick, xticklabels)
plt.ylim(0, 50)
plt.xlabel('Frequency(kHz)')
plt.ylabel('Magnitude(dB)')
plt.title('Frequency Spectrum')
# fig.tight_layout()
# plt.show()
# plt.savefig(os.path.join(figure_path, str1 + '.png'))
# plt.cla()  # Clear axis
# plt.clf()  # Clear figure
# plt.close()  # Close a figure window

plt.subplot(425)
plt.plot(time_line, S_[:, 1].T)
plt.ylabel('Amplitude(V)')

"""plot signal at frequency domain"""
plt.subplot(426)
xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
plt.stem(freq_1, freq_spec_1, markerfmt='None', basefmt='None')
plt.xticks(xtick, xticklabels)
plt.ylim(0, 50)
plt.xlabel('Frequency(kHz)')
plt.ylabel('Magnitude(dB)')
# plt.title('Frequency Spectrum')

plt.subplot(427)
plt.plot(time_line, S_[:, 2].T)
plt.xlabel('Time(sec)')
plt.ylabel('Amplitude(V)')

"""plot signal at frequency domain"""
plt.subplot(428)
xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
plt.stem(freq_2, freq_spec_2, markerfmt='None', basefmt='None')
plt.xticks(xtick, xticklabels)
plt.ylim(0, 50)
plt.xlabel('Frequency(kHz)')
plt.ylabel('Magnitude(dB)')
# plt.title('Frequency Spectrum')
fig.tight_layout()
# plt.show()
plt.savefig(os.path.join(figure_path, str1 + '.png'))
plt.cla()  # Clear axis
plt.clf()  # Clear figure
plt.close()  # Close a figure window
