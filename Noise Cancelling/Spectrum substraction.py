import os
import numpy as np
import pandas as pd
from ipywidgets import Audio
from matplotlib import pyplot as plt
from nptdms.tdmsinfo import display
from scipy import signal
from scipy.io import wavfile


# 读取初始信号 mix.wav
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

f, t, transferred_signal = signal.stft(split_signal, sampling_rate, nperseg=1000)
plt.pcolormesh(t, f, np.abs(transferred_signal), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


module_tf_xmat = abs(transferred_signal)
angle_tf_xmat = np.angle(transferred_signal)
module_tf_bruit = module_tf_xmat[:, 0]

module_reconstruit = np.zeros(module_tf_xmat.shape)
for n in range(module_tf_xmat.shape[1]):
    module_reconstruit[:, n] = module_tf_xmat[:, n] - module_tf_bruit
module_reconstruit[module_reconstruit < 0] = 0

# 将相位和降噪后的幅值重构复信号的频域分布
tf_reconstruit = np.zeros(module_tf_xmat.shape, dtype=complex)
for i in range(module_tf_xmat.shape[0]):
    for j in range(module_tf_xmat.shape[1]):
        tf_reconstruit[i, j] = module_reconstruit[i, j] * np.exp(angle_tf_xmat[i, j] * 1j)

# 使用短时傅立叶变换逆变换重构时域内的信号
_, xrec = signal.istft(tf_reconstruit, sampling_rate)
f, t, filtered_signal = signal.stft(xrec, sampling_rate, nperseg=1024)
plt.pcolormesh(t, f, np.abs(filtered_signal), shading='gouraud')
plt.title('Filtered signal')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
