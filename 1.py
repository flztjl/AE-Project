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
filename = 'Y 267.43 sec to 267.54-4th-test-40mph-1st day@1.csv'

figure_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/100 kHz/Defect/copy/4th-1st day/plot1/'
# '/media/lj/MachineLearning/AE recognition/Data/HTL data/11th/defect/plot1/'
#
os.makedirs(figure_path, exist_ok=True)
split_file = pd.read_csv(os.path.join(file_path, filename))
# acoustic_data = list(split_file.iloc[:, 1])
acoustic_data = split_file.iloc[:, 1].to_numpy()
acoustic_data = acoustic_data[38000: 42000]
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
plt.imshow(logspec_total, extent=[0, 0.03, 0, 500], cmap='jet', aspect='auto', origin='lower')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
fig.tight_layout()
plt.savefig(os.path.join(figure_path, fig_title + '-spectrum.png'))
plt.cla()  # Clear axis
plt.clf()  # Clear figure
plt.close()  # Close a figure window