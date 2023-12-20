import IPython
from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import io

# matplotlib inline

response = urllib.request.urlopen(url)
data, rate = sf.read(io.BytesIO(response.read()))
data = data

# Stationary remove noise
reduced_noise = nr.reduce_noise(y=data, sr=rate, n_std_thresh_stationary=1.5, stationary=True)

# Non-stationary noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate, thresh_n_mult_nonstationary=2, stationary=False)

# ensure that noise reduction does not cause distortion when prop_decrease == 0
noise_reduced = nr.reduce_noise(y=data, sr=rate, prop_decrease=0, stationary=False)

