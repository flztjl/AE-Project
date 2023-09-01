import librosa
import numpy as np
from scipy import signal
from mayavi import mlab

# Generate some example data
t = np.linspace(0, 1, 1000, endpoint=False)
x = signal.square(2 * np.pi * 5 * t)
frequencies = np.arange(1, 100, 2)
wavelet = signal.morlet2(50, 0.8, w=5.0)
wp = signal.cwt(x, signal.morlet2, frequencies)

# Plot the wavelet packet transform
mlab.mesh(np.log2(frequencies), np.arange(wp.shape[1]), np.abs(wp), colormap="jet")
mlab.axes(xlabel='Frequency (log2)', ylabel='Node', zlabel='Amplitude')
mlab.show()
