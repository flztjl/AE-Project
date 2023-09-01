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

import numpy as np
from pywt import wavedecn, waverecn

# Create a 3D array
data = np.random.rand(4, 4, 4)

# Perform the 3D wavelet transform
coeffs = wavedecn(data, 'db1')

# Display the coefficients
print(coeffs)

# Reconstruct the original array
data_recon = waverecn(coeffs, 'db1')

# Compare the original and reconstructed arrays
print(np.allclose(data, data_recon))
