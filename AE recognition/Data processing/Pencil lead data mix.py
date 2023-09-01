import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

defect_data_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/defect/'
noise_data_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/noise ' \
                  'source/'
signal_to_noise_ratio = 0.01

defect_list = os.listdir(defect_data_path)
noise_list = os.listdir(noise_data_path)
for i in range(0, len(defect_list)):
    defect_name = defect_list[i]
    noise_name = noise_list[i]
    filename = defect_name.split('.csv')[0]
    defect_file = pd.read_csv(os.path.join(defect_data_path, defect_name))
    noise_file = pd.read_csv(os.path.join(noise_data_path, noise_name))
    acoustic_data = defect_file.iloc[:, 1]
    noise_data = noise_file.iloc[:, 1]
    simulated_defect = acoustic_data * signal_to_noise_ratio + noise_data
    output_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/duplicated ' \
                  'defect-snr0.01/'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'plot/'), exist_ok=True)
    simulated_defect.to_csv(os.path.join(output_path, filename + '.csv'))

    # pt_step = 1 / 1000000
    # time_sequence = np.arange(0, 0.05, pt_step)
    # fig, ax = plt.subplots(figsize=(6, 3))
    # # fig = plt.figure(figsize=(6, 3))
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(time_sequence[0: 50000], list(noise_data), 'b')
    # plt.plot(time_sequence[0: 50000], list(acoustic_data * signal_to_noise_ratio), 'r')
    # # plt.ylim(-1, 1)
    # plt.title(filename)
    # plt.xlabel('Time(sec)')
    # plt.ylabel('Amplitude(V)')
    #
    # """calculate FFT"""
    # freq_spec_1 = abs(fft(list(acoustic_data * signal_to_noise_ratio)))
    # freq_spec_2 = abs(fft(list(noise_data)))
    # freq_spec_1 = 20 * np.log10(freq_spec_1)
    # freq_spec_2 = 20 * np.log10(freq_spec_2)
    # num_datapts = int(np.fix(len(freq_spec_1) / 2))
    # freq_spec_1 = freq_spec_1[0: num_datapts]
    # freq_spec_2 = freq_spec_2[0: num_datapts]
    # fft_pts = np.arange(0, num_datapts)
    # freq1 = fft_pts * (1000000 / (2 * num_datapts))
    #
    # """plot signal at frequency domain"""
    # plt.subplot(212)
    # xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
    # xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
    # plt.stem(freq1, freq_spec_2, markerfmt='None', basefmt='None', linefmt='blue')
    # plt.stem(freq1, freq_spec_1, markerfmt='None', basefmt='None', linefmt='red')
    # plt.xticks(xtick, xticklabels)
    # plt.ylim(0, 50)
    # plt.xlabel('Frequency(kHz)')
    # plt.ylabel('Magnitude(dB)')
    # plt.title('Frequency Spectrum')
    #
    # fig.tight_layout()
    # plt.savefig(os.path.join(output_path, 'plot/' + filename + '.png'))
    # plt.cla()  # Clear axis
    # plt.clf()  # Clear figure
    # plt.close()  # Close a figure window


