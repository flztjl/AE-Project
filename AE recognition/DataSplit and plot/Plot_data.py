import os
import librosa
import noisereduce as nr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.fftpack import fft

output_path_sensor = ('/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/optical '
                      'test/newtest2/test1/')
figure_path = os.path.join(output_path_sensor, 'collection')
os.makedirs(figure_path, exist_ok=True)
split_file = pd.read_csv(os.path.join(output_path_sensor, '72-1.csv'))
acoustic_data = list(split_file.iloc[:, 1])
# x = data[':,1']
calibration = np.mean(acoustic_data)
acoustic_data -= calibration
sampling_rate = 1000000

xpt_start = 0.64
xpt_end = 0.66
split_signal = acoustic_data[int(np.fix(xpt_start * sampling_rate)): int(np.fix(xpt_end * sampling_rate))]

"""Stationary remove noise"""
reduced_noise = nr.reduce_noise(y=split_signal, sr=sampling_rate, freq_mask_smooth_hz=2000, n_std_thresh_stationary=1.5,
                                stationary=True)

"""Non-stationary noise reduction"""
reduced_noise2 = nr.reduce_noise(y=split_signal, sr=sampling_rate, freq_mask_smooth_hz=2000,
                                 thresh_n_mult_nonstationary=5, stationary=False)

""" ensure that noise reduction does not cause distortion when prop_decrease == 0"""
reduced_noise3 = nr.reduce_noise(y=split_signal, sr=sampling_rate, freq_mask_smooth_hz=2000, prop_decrease=0,
                                 stationary=False)

# b, a = signal.butter(3, 0.06, 'highpass')  # setup filter parameters
# split_signal = signal.filtfilt(b, a, split_signal)  # data is signal to be filtered

pt_step = 1
# str1 = 'Signal ' + str('%.2f' % xpt_start) + ' to ' + str(xpt_end) + ' sec'
fig_title = 'Signal ' + str(xpt_start) + ' to ' + str(xpt_end) + ' sec'
time_sequence = np.arange(float(xpt_start) * sampling_rate, float(xpt_end) * sampling_rate, pt_step)

fig = plt.figure(figsize=(6, 3))
plt.figure(1)
plt.subplot(211)
plt.plot(np.round(time_sequence / sampling_rate, 2), split_signal)
# plt.ylim(-5, 5)
plt.title(fig_title)
plt.xlabel('Time(sec)')
plt.ylabel('Amplitude(V)')

"""calculate FFT"""
freq_spec = abs(fft(split_signal))
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
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
fig.tight_layout()
plt.savefig(os.path.join(figure_path, fig_title + '.png'))
plt.cla()  # Clear axis
plt.clf()  # Clear figure
plt.close()  # Close a figure window

fig, ax = plt.subplots(figsize=(6, 4))
# fig = plt.figure(figsize=(6, 3))
plt.figure(2)
# widths = np.arange(1, 31)
# cwtmatr = signal.cwt(acoustic_data, signal.ricker, widths)
# plt.imshow(cwtmatr, extent=[0, 0.11, 1, 31], cmap='jet', aspect='auto',
#            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
_, _, sgram = signal.spectrogram(split_signal, sampling_rate, nperseg=256, scaling='spectrum')
logspec_total = librosa.amplitude_to_db(sgram)
# librosa.display.specshow(logspec_total, sr=sampling_rate, x_axis='time', y_axis='linear', cmap='jet')
plt.imshow(logspec_total, extent=[0, 0.05, 0, 500000], cmap='jet', aspect='auto', origin='lower')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
fig.tight_layout()
plt.savefig(os.path.join(figure_path, fig_title + '-spectrum.png'))
plt.cla()  # Clear axis
plt.clf()  # Clear figure
plt.close()  # Close a figure window

# """Wavelet packet analysis"""
# """Print mel-spectrum figures"""
# def wpd_plt(y, wave='sym5', n=None, best_basis=None):
#     # wpd decompose
#     wp = pywt.WaveletPacket(data=y, wavelet=wave, mode='symmetric', maxlevel=n)
#     n = wp.maxlevel
#     # Calculate the coefficients for each node, where map Medium, key by'aa'Wait, value For List
#     map = {}
#     map[1] = y
#     for row in range(1, n + 1):
#         lev = []
#         for i in [node.path for node in wp.get_level(row, 'freq')]:
#             map[i] = wp[i].data
#
#     # Mapping
#     plt.figure(figsize=(15, 10))
#     ax = plt.subplot(n + 1, 1, 1)  # Draw the first graph
#     ax.set_title('data')
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     plt.plot(map[1])
#     for i in range(2, n + 2):
#         level_num = pow(2, i - 1)  # Starting with the second row, the power of 2 of the previous row is calculated
#         # Getting decomposed at each level node: For example, the third layer['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
#         re = [node.path for node in wp.get_level(i - 1, 'freq')]
#         for j in range(1, level_num + 1):
#             if best_basis != None:
#                 if True in [(re[j - 1].startswith(b) & (re[j - 1] != b)) for b in best_basis]:
#                     continue
#             ax = plt.subplot(n + 1, level_num, level_num * (i - 1) + j)
#             plt.plot(map[re[j - 1]])  # List starts at 0
#             ax.set_title(re[j - 1])
#             ax.set_yticklabels([])
#             ax.set_xticklabels([])
#     plt.show()
