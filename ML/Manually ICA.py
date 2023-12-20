import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt
import seaborn as sns

np.random.seed(0)
sns.set(rc={'figure.figsize': (11.7, 8.27)})

"""define g and g’ which we’ll use to determine the new value for w"""


def g(x):
    return np.tanh(x)


def g_der(x):
    return 1 - g(x) * g(x)


"""create a function to center the signal by subtracting the mean"""


def center(X):
    X = np.array(X)

    mean = X.mean(axis=1, keepdims=True)

    return X - mean


"""define a function to whiten the signal using the method described above"""


def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten


"""define a function to update the de-mixing matrix w"""


def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new


"""Finally, we define the main method which calls the preprocessing functions, 
initializes w to some random set of values and iteratively updates w. Again, 
convergence can be judged by the fact that an ideal w would be orthogonal, 
and hence w multiplied by its transpose would be approximately equal to 1. 
After computing the optimal value of w for each component, 
we take the dot product of the resulting matrix and the signal x to get the sources."""


def ica(X, iterations, tolerance=1e-5):
    X = center(X)
    X = whitening(X)
    components_nr = X.shape[0]
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)
    for i in range(components_nr):
        w = np.random.rand(components_nr)
        for j in range(iterations):
            w_new = calculate_new_w(w, X)
            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            w = w_new
            if distance < tolerance:
                break
        W[i, :] = w
    S = np.dot(W, X)
    return S


"""define a function to plot and compare the original, mixed and predicted signals"""


def plot_mixture_sources_predictions(X, original_sources, S):
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    for x in X:
        plt.plot(x)
    plt.title("mixtures")
    plt.subplot(3, 1, 2)
    for s in original_sources:
        plt.plot(s)
    plt.title("real sources")
    plt.subplot(3, 1, 3)
    for s in S:
        plt.plot(s)
    plt.title("predicted sources")
    fig.tight_layout()
    plt.show()


split_file_1 = pd.read_csv('D:/Python/HTL/1st day/1st test-20mph-sensor1/1st test-20mph-3/23-1.csv')
split_file_2 = pd.read_csv('D:/Python/HTL/1st day/1st test-20mph-sensor2/1st test-20mph-3/23-2.csv')
split_file_3 = pd.read_csv('D:/Python/HTL/1st day/1st test-20mph-sensor3/1st test-20mph-3/23-3.csv')

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
# x = data[':,1']

sampling_rate = 1000000

i = 0.785
j = 0.795
time_stamp = 423 + i

split_signal_1 = acoustic_data_1[int(i * sampling_rate): int(j * sampling_rate)]
split_signal_2 = acoustic_data_2[int(i * sampling_rate): int(j * sampling_rate)]
split_signal_3 = acoustic_data_3[int(i * sampling_rate): int(j * sampling_rate)]

# hx = fftpack.hilbert(split_signal)
# hy = np.abs(hx)
# hy = np.sqrt(split_signal**2+hx**2)
xpt_start = i
xpt_end = j
pt_step = 1 / sampling_rate
str1 = 'Signal ' + str('%.2f' % xpt_start) + ' to ' + str(xpt_end) + ' sec'
str2 = 'Signal ' + str('%.2f' % time_stamp) + ' to ' + str(time_stamp - i + j) + ' sec'
time_sequence = np.arange(float(xpt_start), float(xpt_end), pt_step)
time_line = time_sequence[0: int(sampling_rate * (j - i))]

refined_signal = np.c_[split_signal_1, split_signal_2, split_signal_3]

# refined_signal = refined_signal.T
S = ica(refined_signal, iterations=2000)

fig = plt.figure()
plt.figure(1)
plt.subplot(411)
plt.plot(time_line, refined_signal[1, :])
plt.title(str1)
plt.ylabel('Amplitude(V)')

plt.subplot(412)
plt.plot(time_line, S[0, :])
plt.ylabel('Amplitude(V)')

plt.subplot(413)
plt.plot(time_line, S[1, :])
plt.ylabel('Amplitude(V)')

plt.subplot(414)
plt.plot(time_line, S[2, :])
plt.xlabel('Time(sec)')
plt.ylabel('Amplitude(V)')

fig.tight_layout()
plt.show()
