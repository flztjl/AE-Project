# To create a 3D spectrum plot similar to the one in the provided image, we will perform a Fast Fourier Transform (FFT)
# on the sensor data to transform it from time domain to frequency domain.

import numpy as np

# Perform FFT on the sensor data
fft_values = np.fft.fft(data['sensor1'].values)
fft_freq = np.fft.fftfreq(data['sensor1'].index.size, d=1e-6)  # d is the sample spacing

# Only take the positive half of the spectrum and corresponding frequencies
pos_mask = np.where(fft_freq > 0)
freqs = fft_freq[pos_mask]
spectrum = np.abs(fft_values[pos_mask])

# Prepare the data for a 3D plot
# We are assuming time on one axis and frequency on another, with the amplitude as the height (z-axis).
X, Y = np.meshgrid(data['Time'], freqs)
Z = np.tile(spectrum, (len(data['Time']), 1)).T  # Duplicate the spectrum for each time point

# Now let's create the 3D plot
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Use a surface plot for the 3D spectrum
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Time [s]')
ax.set_ylabel('Frequency [kHz]')
ax.set_zlabel('Amplitude')

# Set the view angle to match the provided screenshot angle
ax.view_init(30, 120)

plt.show()
