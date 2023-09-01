import os
import functools
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fftpack import fft


def decorrstretch(A, tol=None):
    """
    Apply decorrelation stretch to image
    Arguments:
    A   -- image in cv2/numpy.array format
    tol -- upper and lower limit of contrast stretching
    """

    # save the original shape
    orig_shape = A.shape
    # reshape the image
    #         B G R
    # pixel 1 .
    # pixel 2   .
    #  . . .      .
    A = A.reshape((-1, 3)).astype(np.float)
    # covariance matrix of A
    cov = np.cov(A.T)
    # source and target sigma
    sigma = np.diag(np.sqrt(cov.diagonal()))
    # eigen decomposition of covariance matrix
    eigval, V = np.linalg.eig(cov)
    # stretch matrix
    S = np.diag(1 / np.sqrt(eigval))
    # compute mean of each color
    mean = np.mean(A, axis=0)
    # substract the mean from image
    A -= mean
    # compute the transformation matrix
    T = functools.reduce(np.dot, [sigma, V, S, V.T])
    # compute offset
    offset = mean - np.dot(mean, T)
    # transform the image
    A = np.dot(A, T)
    # add the mean and offset
    A += mean + offset
    # restore original shape
    B = A.reshape(orig_shape)
    # for each color...
    for b in range(3):
        # apply contrast stretching if requested
        if tol:
            # find lower and upper limit for contrast stretching
            low, high = np.percentile(B[:, b], 100 * tol), np.percentile(B[:, b], 100 - 100 * tol)
            B[B < low] = low
            B[B > high] = high
        # ...rescale the color values to 0..255
        B[:, b] = 255 * (B[:, b] - B[:, b].min()) / (B[:, b].max() - B[:, b].min())
    # return it as uint8 (byte) image
    return B.astype(np.uint8)


split_file_1 = pd.read_csv('D:/Python/HTL/1st day/1st test-20mph-sensor1/1st test-20mph-3/67-1.csv')
split_file_2 = pd.read_csv('D:/Python/HTL/1st day/1st test-20mph-sensor2/1st test-20mph-3/66-2.csv')
split_file_3 = pd.read_csv('D:/Python/HTL/1st day/1st test-20mph-sensor3/1st test-20mph-3/66-3.csv')

acoustic_data_1 = list(split_file_1.iloc[:, 1])
calibration_1 = np.mean(acoustic_data_1)
acoustic_data_1 -= calibration_1

acoustic_data_2 = list(split_file_2.iloc[:, 1])
calibration_2 = np.mean(acoustic_data_2)
acoustic_data_2 -= calibration_2

acoustic_data_3 = list(split_file_3.iloc[:, 1])
calibration_3 = np.mean(acoustic_data_3)
acoustic_data_3 -= calibration_3

figure_path = os.path.join('D:/Python/HTL/1st day/Defects/1st test/potential defects/')
os.makedirs(figure_path, exist_ok=True)

sampling_rate = 1000000

i1 = 0.28
j1 = 0.38
k1 = math.ceil(sampling_rate * (j1 - i1))
time_stamp = 667 + i1

i2 = 0.07
j2 = 0.17
k2 = math.ceil(sampling_rate * (j2 - i2))

i3 = 0.67
j3 = 0.77
k3 = math.ceil(sampling_rate * (j3 - i3))

split_signal_1 = acoustic_data_1[int(i1 * sampling_rate): int(j1 * sampling_rate)]
split_signal_2 = acoustic_data_2[int(i2 * sampling_rate): int(j2 * sampling_rate)]
split_signal_3 = acoustic_data_3[int(i3 * sampling_rate): int(j3 * sampling_rate)]

"""Band-pass filter"""
b1, a1 = signal.butter(18, 0.16, 'highpass')  # setup filter parameters
split_signal_1 = signal.filtfilt(b1, a1, split_signal_1)  # data is signal to be filtered

"""Band-pass filter"""
b2, a2 = signal.butter(18, 0.16, 'highpass')  # setup filter parameters
split_signal_2 = signal.filtfilt(b2, a2, split_signal_2)  # data is signal to be filtered

"""Band-pass filter"""
b3, a3 = signal.butter(18, 0.16, 'highpass')  # setup filter parameters
split_signal_3 = signal.filtfilt(b3, a3, split_signal_3)  # data is signal to be filtered

xpt_start = i1
xpt_end = j1
pt_step = 1 / sampling_rate / 2
str1 = 'Signal ' + str('%.2f' % time_stamp) + ' to ' + str(time_stamp + j1 - i1) + ' sec'
str2 = 'Signal ' + str('%.2f' % time_stamp) + ' to ' + str(time_stamp + j1 - i1) + ' sec-1'
time_sequence = np.arange(float(xpt_start), float(xpt_end), pt_step)
time_line = time_sequence[0: k1]
time_line1 = time_sequence[0: k1*2-1]

refined_signal = np.c_[split_signal_1, split_signal_2, split_signal_3]

ica = signal.correlate(split_signal_1, split_signal_2)
ica1 = signal.correlate(split_signal_1, split_signal_3)
S_ = ica #.fit_transform(refined_signal)
S_1 = ica1

"""calculate FFT"""
# freq_spec_0 = abs(fft(S_.T))
# freq_spec_0 = 20 * np.log10(freq_spec_0)
# num_datapts = int(np.fix(len(freq_spec_0) / 2))
# freq_spec_0 = freq_spec_0[0: num_datapts]
# fft_pts = np.arange(0, num_datapts)
# freq_0 = fft_pts * (sampling_rate / (2 * num_datapts))

"""calculate FFT"""
# freq_spec_1 = abs(fft(S_[:, 1].T))
# freq_spec_1 = 20 * np.log10(freq_spec_1)
# num_datapts = int(np.fix(len(freq_spec_1) / 2))
# freq_spec_1 = freq_spec_1[0: num_datapts]
# fft_pts = np.arange(0, num_datapts)
# freq_1 = fft_pts * (sampling_rate / (2 * num_datapts))

"""calculate FFT"""
# freq_spec_2 = abs(fft(S_[:, 2].T))
# freq_spec_2 = 20 * np.log10(freq_spec_2)
# num_datapts = int(np.fix(len(freq_spec_2) / 2))
# freq_spec_2 = freq_spec_0[0: num_datapts]
# fft_pts = np.arange(0, num_datapts)
# freq_2 = fft_pts * (sampling_rate / (2 * num_datapts))

fig = plt.figure()
plt.figure(1)
plt.subplot(311)
plt.plot(time_line, refined_signal[:, 0])
plt.title(str1)
plt.ylabel('Amplitude(V)')

plt.subplot(312)
plt.plot(time_line, refined_signal[:, 1])
plt.ylabel('Amplitude(V)')

plt.subplot(313)
plt.plot(time_line, refined_signal[:, 2])
plt.ylabel('Amplitude(V)')

fig.tight_layout()
plt.savefig(os.path.join(figure_path, str1 + '.png'))
plt.cla()  # Clear axis
plt.clf()  # Clear figure
plt.close()

plt.figure(2)
plt.subplot(211)
plt.plot(time_line1, S_.T)
plt.ylabel('Amplitude(V)')

plt.subplot(212)
plt.plot(time_line1, S_1.T)
plt.ylabel('Amplitude(V)')

"""plot signal at frequency domain"""
# plt.subplot(313)
# xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
# xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
# plt.stem(freq_0, freq_spec_0, markerfmt='None', basefmt='None')
# plt.xticks(xtick, xticklabels)
# plt.ylim(0, 50)
# plt.xlabel('Frequency(kHz)')
# plt.ylabel('Magnitude(dB)')
# plt.title('Frequency Spectrum')
fig.tight_layout()
plt.savefig(os.path.join(figure_path, str1 + '-1.png'))
plt.cla()  # Clear axis
plt.clf()  # Clear figure
plt.close()  # Close a figure window

# plt.figure(2)
# plt.subplot(211)
# plt.plot(time_line, S_[:, 1].T)
# plt.ylabel('Amplitude(V)')

"""plot signal at frequency domain"""
# plt.subplot(212)
# xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
# xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
# plt.stem(freq_1, freq_spec_1, markerfmt='None', basefmt='None')
# plt.xticks(xtick, xticklabels)
# plt.ylim(0, 50)
# plt.xlabel('Frequency(kHz)')
# plt.ylabel('Magnitude(dB)')
# plt.title('Frequency Spectrum')

# plt.subplot(413)
# plt.plot(time_line, S_[:, 2].T)
# plt.xlabel('Time(sec)')
# plt.ylabel('Amplitude(V)')
#
# """plot signal at frequency domain"""
# plt.subplot(414)
# xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
# xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
# plt.stem(freq_2, freq_spec_2, markerfmt='None', basefmt='None')
# plt.xticks(xtick, xticklabels)
# plt.ylim(0, 50)
# plt.xlabel('Frequency(kHz)')
# plt.ylabel('Magnitude(dB)')
# plt.title('Frequency Spectrum')
# fig.tight_layout()
# plt.savefig(os.path.join(figure_path, str2 + '.png'))
# plt.cla()  # Clear axis
# plt.clf()  # Clear figure
# plt.close()  # Close a figure window
