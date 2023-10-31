import librosa
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from scipy import signal
# import pywt


# def extract_integer(filename):
#     if filename != 'plot':
#         input_order = filename.split('-')[0]
#         return int(input_order)

def signal_processing(output_path_sensor, sensor_num):
    """cancel data shift and read split file"""
    dir_list = os.listdir(output_path_sensor)  # dir is your directory path
    # dir_list = sorted(os.listdir(output_path_sensor), key=extract_integer)
    for i in range(72, 200): # len(dir_list) - 1
        # if filename == 'plot':
        #     continue
        # if filename == dir_list[len(dir_list) - 1]:
        #     break
        # fold_number = output_path_sensor.split('/')[8][4]
        # start_time = filename.split('-')[0]
        figure_path = os.path.join(output_path_sensor, 'plot')
        os.makedirs(figure_path, exist_ok=True)
        split_file = pd.read_csv(os.path.join(output_path_sensor, str(i) + '-' + str(sensor_num) + '.csv'))
        acoustic_data = list(split_file.iloc[:, 1])
        # x = data[':,1']
        calibration = np.mean(acoustic_data)
        acoustic_data -= calibration
        sampling_rate = 1000000
        count = 20

        """Wavelet packet analysis"""
        # dec_lv = 8
        # total_scales = 2 ** dec_lv
        # data_matrix = []
        # data_dec = pywt.WaveletPacket(data=acoustic_data, wavelet='db1', mode='symmetric', maxlevel=dec_lv)
        # for j in [node.path for node in data_dec.get_level(dec_lv, 'freq')]:
        #     data_array = data_dec[j].data
        #     data_matrix = np.append(data_matrix, data_array)
        # data_matrix = np.reshape(data_matrix, (total_scales, -1))
        # # data_matrix[0:40] = 0
        # # data_matrix[153:256] = 0
        # xlim = data_matrix.shape[1]
        # x_step = xlim / 10
        #
        # plt.figure(1)
        # plt.pcolor(abs(data_matrix), cmap='Blues')
        # plt.colorbar()
        # xtick = np.arange(0, xlim, x_step)
        # xticklabels = np.arange(float(i), float(i + 1), 0.1)
        # xticklabels = np.around(xticklabels, decimals=1)
        # plt.xticks(xtick, xticklabels, rotation=-90)
        # plt.ylabel("freq(kHz)")
        # plt.xlabel("time(s)")
        # plt.subplots_adjust(hspace=0.4)
        # plt.savefig(os.path.join(figure_path, str(i) + '.png'))
        # plt.cla()  # Clear axis
        # plt.clf()  # Clear figure
        # plt.close()  # Close a figure window

        """FFT calculation"""
        """Split each file into time units, and implement high-pass filter"""
        for j in range(0, count):
            split_signal = acoustic_data[
                           int(np.fix(j / count * sampling_rate)): int(np.fix((j + 1) / count * sampling_rate))]
            xpt_start = (j / count) + i
            xpt_end = (j + 1) / count + i
            pt_step = 1 / sampling_rate
            # str1 = 'Signal ' + str('%.2f' % xpt_start) + ' to ' + str(xpt_end) + ' sec'
            fig_title = 'Signal ' + "{:.2f}".format(xpt_start) + ' to ' + str(xpt_end) + ' sec'
            time_sequence = np.arange(float(xpt_start), float(xpt_end), pt_step)
            print(xpt_start, xpt_end, pt_step)  # prev_data = np.ones(int(sampling_rate / (2 * count)))

            """plot filtered signal at time domain"""
            """Band-pass filter"""
            # b, a = signal.butter(3, 0.2, 'highpass')  # setup filter parameters
            # acoustic_data_filtered = signal.filtfilt(b, a, split_signal)  # data is signal to be filtered

            # acoustic_data_filtered = split_signal

            fig = plt.figure(figsize=(6, 3))
            plt.figure(1)
            plt.subplot(211)
            plt.plot(time_sequence[0: 50000], split_signal)
            # plt.ylim(-5, 5)
            plt.title(fig_title)
            plt.xlabel('Time(sec)')
            plt.ylabel('Amplitude(V)')

            """calculate FFT"""
            freq_spec = abs(fft(split_signal))
            # freq_spec = 20 * np.log10(freq_spec)
            num_datapts = int(np.fix(len(freq_spec) / 2))
            freq_spec = freq_spec[0: num_datapts]
            fft_pts = np.arange(0, num_datapts)
            freq1 = fft_pts * (sampling_rate / (2 * num_datapts))

            """plot signal at frequency domain"""
            plt.subplot(212)
            xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
            xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
            plt.stem(freq1, freq_spec, markerfmt='None', basefmt='None')
            plt.xticks(xtick, xticklabels)
            plt.ylim(0, 50)
            plt.xlabel('Frequency(kHz)')
            plt.ylabel('Magnitude(dB)')
            plt.title('Frequency Spectrum')
            plt.tight_layout()

            # plt.subplot(212)
            # plt.plot(time_sequence, acoustic_data_filtered)
            # plt.ylim(-5, 5)
            # plt.ylabel('Amplitude(V)')

            fig.tight_layout()
            plt.savefig(os.path.join(figure_path, fig_title + '.png'))
            plt.cla()  # Clear axis
            plt.clf()  # Clear figure
            plt.close()  # Close a figure window

            fig, ax = plt.subplots(figsize=(6, 4))
            # fig = plt.figure(figsize=(6, 3))
            plt.figure(2)
            # widths = np.arange(1, 31)
            # cwtmatr = signal.cwt(acoustic_data, signal.ricker, widths)
            # plt.imshow(cwtmatr, extent=[0, 0.11, 1, 31], cmap='jet', aspect='auto',
            #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
            _, _, sgram = signal.spectrogram(split_signal, sampling_rate, nperseg=256, scaling='spectrum')
            logspec_total = librosa.amplitude_to_db(sgram)
            # librosa.display.specshow(logspec_total, sr=sampling_rate, x_axis='time', y_axis='linear', cmap='jet')
            plt.imshow(logspec_total, extent=[0, 0.05, 0, 500], cmap='jet', aspect='auto', origin='lower')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            fig.tight_layout()
            plt.savefig(os.path.join(figure_path, fig_title + '-spectrum.png'))
            plt.cla()  # Clear axis
            plt.clf()  # Clear figure
            plt.close()  # Close a figure window
