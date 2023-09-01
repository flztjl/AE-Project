from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

b, a = signal.butter(10, 0.06, 'high')
w, h = signal.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response-order 10')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.ylim(-6, 6)
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()
