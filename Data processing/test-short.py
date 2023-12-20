import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from matplotlib.patches import Rectangle

file = '/media/lj/AE Project/Archives/HTL data/1st trial/testing fold/Y 238.18 sec to 238.29 sec-3.csv'
# sound_clip, sr = librosa.load(file)
detection_result = pd.read_csv(file)
sound_clip = detection_result.iloc[:, 1]
b, a = signal.butter(2, 30000, 'highpass', fs=1000000)  # setup filter parameters
sound_clip = signal.filtfilt(b, a, sound_clip)  # data is signal to be filtered

# sound_clip = sound_clip.to_numpy()


"""Print mel-spectrum figures"""
fig, ax = plt.subplots()
plt.figure(1)
melspec_total = librosa.feature.melspectrogram(sound_clip, n_mels=64)
logspec_total = librosa.amplitude_to_db(melspec_total)
librosa.display.specshow(logspec_total, cmap='jet', sr=1000000, x_axis='time', y_axis='fft', fmax=500000)
plt.colorbar(format='%+2.0f dB')
fig.tight_layout()
plt.savefig(os.path.join('/media/lj/AE Project/Archives/UrbanSound8K/audio/1.png'))
plt.cla()  # Clear axis
plt.clf()  # Clear figure
plt.close()  # Close a figure window
