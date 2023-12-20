import os
import matplotlib.pylab as plt
import padasip as pa
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from scipy import signal

split_file = pd.read_csv('D:/Python/RDTF/5 mph/1st test reversed/66-1.csv')
# D:/Python/HTL/1st day/1st test-20mph-3/153
# figure_path = os.path.join('D:/Python/RDTF/5 mph/1st test reversed/defects-refined/1/')
# os.makedirs(figure_path, exist_ok=True)
acoustic_data = list(split_file.iloc[:, 1])
# x = data[':,1']
calibration = np.mean(acoustic_data)
acoustic_data -= calibration
sampling_rate = 1000000

time_stamp = 66
i = 0.625
j = 0.635

split_signal_copy = acoustic_data[int(i * sampling_rate): int(j * sampling_rate)]
split_signal = np.reshape(acoustic_data[int(i * sampling_rate): int(j * sampling_rate)], (-1, 1))
noise_sample_copy = acoustic_data[int((i-0.01) * sampling_rate): int((j-0.01) * sampling_rate)]
noise_sample = np.reshape(acoustic_data[int((i-0.01) * sampling_rate): int((j-0.01) * sampling_rate)], (-1, 1))
noise_reference = split_signal - noise_sample

# creation of data
# N = 500
# x = np.random.normal(0, 1, (N, 4))  # input matrix
# v = np.random.normal(3, 3, N)  # noise
# z = 2 * x[:, 0] + 0.1 * x[:, 1] - 4 * x[:, 2] + 0.5 * x[:, 3]
# d = 2 * x[:, 0] + 0.1 * x[:, 1] - 4 * x[:, 2] + 0.5 * x[:, 3] + v  # target

"""Adaptive filter"""
f = pa.filters.FilterLMS(n=1, mu=0.001, w="random")
filtered_signal, e, w = f.run(split_signal, noise_reference)

"""calculate FFT of original signal"""
freq_spec_original = abs(fft(split_signal_copy))
# freq_spec_original = 20 * np.log10(freq_spec_original)
num_datapts_original = int(np.fix(len(freq_spec_original) / 2))
freq_spec_original = freq_spec_original[0: num_datapts_original]
fft_pts_original = np.arange(0, num_datapts_original)
freq_original = fft_pts_original * (sampling_rate / (2 * num_datapts_original))

"""calculate FFT of filtered signal"""
freq_spec_filtered = abs(fft(filtered_signal))
num_datapts_filtered = int(np.fix(len(freq_spec_filtered) / 2))
freq_data_filtered = freq_spec_filtered[0: num_datapts_filtered]
fft_pts_filtered = np.arange(0, num_datapts_filtered)
freq_filtered = fft_pts_filtered * (sampling_rate / (2 * num_datapts_filtered))

# show results
plt.figure(figsize=(15, 9))
plt.subplot(211)
plt.title("Adaptation")
plt.xlabel("samples - k")
plt.plot(split_signal, "b", label="original")

plt.subplot(212)
xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
plt.stem(freq_original, freq_spec_original, markerfmt='None', basefmt='None')
plt.xticks(xtick, xticklabels)
# plt.ylim(0, 50)
plt.xlabel('Frequency(kHz)')
plt.ylabel('Magnitude(dB)')
plt.title('Frequency Spectrum')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 9))
plt.subplot(211)
plt.title("Adaptation")
plt.xlabel("samples - k")
plt.plot(filtered_signal, "g", label="output")
plt.legend()

plt.subplot(212)
xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
plt.stem(freq_filtered, freq_data_filtered, markerfmt='None', basefmt='None')
plt.xticks(xtick, xticklabels)
# plt.ylim(0, 50)
plt.xlabel('Frequency(kHz)')
plt.ylabel('Magnitude(dB)')
plt.title('Frequency Spectrum')
plt.tight_layout()
plt.show()


# plt.title("Filter error")
# plt.xlabel("samples - k")
# plt.plot(10 * np.log10(e ** 2), "r", label="e - error [dB]")
# plt.legend()
# plt.tight_layout()
# plt.show()
