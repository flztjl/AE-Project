import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from matplotlib.patches import Rectangle

sampling_rate = 1000000
figure_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/16th/Defect/TD/plot/'
os.makedirs(figure_path, exist_ok=True)
defect_file_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/16th/Defect/TD/'
# defect_file_path_2 = '/media/lj/MachineLearning/AE recognition/Data/HTL data/11th/shell/'
# noise_file_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/11th/noise/'
detection_result = pd.read_csv('/media/lj/MachineLearning/AE recognition/Data/HTL data/16th/processed/td/1.csv')
filename_list = detection_result.iloc[:, 3]
sr = 1000000
defect_code = 1

for filename in filename_list:
    data_type = filename.split(' ')[0]
    detect_number = list(filename_list).index(filename)
    prediction_result = detection_result.iloc[detect_number, 2]
    # defect_code = detection_result.iloc[detect_number, 4]
    global print_label
    global defect_type

    if defect_code == 0:
        defect_type = 'SHELL'
        if prediction_result == 0:
            print_label = 'False Negative'
        else:
            print_label = 'True Positive'

        if tag == 'Y':
            split_file = pd.read_csv(os.path.join(defect_file_path, filename))
            acoustic_data = split_file.iloc[:, 1]
            acoustic_data = acoustic_data.to_numpy()
            xpt_start = ' '.join(filename.split(' ')[1:7])
            xpt_end = filename.split('-')[1]
            pt_step = 1 / sampling_rate
            fig_title = 'Defect' + xpt_start
            # fig_title = 'Signal ' + str(xpt_start) + ' to ' + str(xpt_end) + ' sec'
            # str1 = 'Signal ' + str('%.2f' % xpt_start) + ' to ' + str(xpt_end) + ' sec'

            # time_sequence = np.arange(float(xpt_start), float(xpt_end), pt_step)
            time_sequence = np.arange(0, 0.11, pt_step)
            defect_start = 0.05

            # fig = plt.figure(figsize=(6, 3))
            plt.figure(1)
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 4))
            plt.subplot(211)
            plt.plot(time_sequence[0: 110000], acoustic_data, zorder=1)
            plt.ylim(-10, 10)
            plt.title(fig_title)
            plt.xlabel('Time(sec)')
            plt.ylabel('Amplitude(V)')
            plt.legend([print_label + ' + ' + defect_type], loc='upper left')
            #rect1 = plt.Rectangle((-0.02, 13), 0.15, 30, fc='none', ec='r', lw=2, zorder=2)
            rect = plt.Rectangle((defect_start, -9.5), 0.01, 19, fc='none', ec='r', lw=2, zorder=2)
            ax0.add_patch(rect)

            """calculate FFT"""
            freq_spec = abs(fft(acoustic_data))
            freq_spec = 20 * np.log10(freq_spec)
            num_datapts = int(np.fix(len(freq_spec) / 2))
            freq_spec = freq_spec[0: num_datapts]
            fft_pts = np.arange(0, num_datapts)
            freq1 = fft_pts * (sampling_rate / (2 * num_datapts))

            """plot signal at frequency domain"""
            plt.subplot(212)
            xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
            xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
            ax1.stem(freq1, freq_spec, markerfmt='None', basefmt='None')
            plt.xticks(xtick, xticklabels)
            # plt.ylim(0, 50)
            plt.xlabel('Frequency(kHz)')
            plt.ylabel('Magnitude(dB)')
            plt.title('Frequency Spectrum')

            fig.tight_layout()
            plt.savefig(os.path.join(figure_path, fig_title + '.png'))
            plt.cla()  # Clear axis
            plt.clf()  # Clear figure
            plt.close()  # Close a figure window

            """Print mel-spectrum figures"""
            fig1, ax = plt.subplots()
            plt.figure(2)
            start_time = filename.split(' ')[1]
            end_time = filename.split(' ')[3].split('-')[0]
            title = 'Defect ' + start_time + ' sec to ' + end_time + ' sec'
            melspec_total = librosa.feature.melspectrogram(acoustic_data, n_mels=512)
            logspec_total = librosa.amplitude_to_db(melspec_total)
            librosa.display.specshow(logspec_total, cmap='jet', sr=sr, x_axis='time', y_axis='linear')
            ax.add_patch(Rectangle((defect_start, 10000), 0.01, 480000, fc='none', ec='r', lw=2, zorder=2))
            plt.colorbar(format='%+2.0f dB')
            plt.title(print_label + ' + ' + defect_type)
            fig1.tight_layout()
            plt.savefig(os.path.join(figure_path, title + 'mel.png'))
            plt.cla()  # Clear axis
            plt.clf()  # Clear figure
            plt.close()  # Close a figure window
        else:
            continue

    if defect_code == 1:
        defect_type = 'TD'
        if prediction_result == 0:
            print_label = 'False Negative'
        else:
            print_label = 'True Positive'

        tag = filename.split('/')[10].split(' ')[0]
        if tag == 'Y':
            split_file = pd.read_csv(os.path.join(defect_file_path, filename))
            acoustic_data = split_file.iloc[:, 1]
            acoustic_data = acoustic_data.to_numpy()
            xpt_start = ' '.join(filename.split(' ')[1:7])
            xpt_end = filename.split('-')[1]
            pt_step = 1 / sampling_rate
            fig_title = 'Defect' + xpt_start
            # fig_title = 'Signal ' + str(xpt_start) + ' to ' + str(xpt_end) + ' sec'
            # str1 = 'Signal ' + str('%.2f' % xpt_start) + ' to ' + str(xpt_end) + ' sec'

            # time_sequence = np.arange(float(xpt_start), float(xpt_end), pt_step)
            time_sequence = np.arange(0, 0.11, pt_step)
            defect_start = 0.05

            # fig = plt.figure(figsize=(6, 3))
            plt.figure(1)
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 4))
            plt.subplot(211)
            plt.plot(time_sequence[0: 110000], acoustic_data, zorder=1)
            plt.ylim(-10, 10)
            plt.title(fig_title)
            plt.xlabel('Time(sec)')
            plt.ylabel('Amplitude(V)')
            plt.legend([print_label + ' + ' + defect_type], loc='upper left')
            rect1 = plt.Rectangle((-0.02, 13), 0.15, 30, fc='none', ec='r', lw=2, zorder=2)
            rect2 = plt.Rectangle((defect_start, -9.5), 0.01, 19, fc='none', ec='r', lw=2, zorder=2)
            ax0.add_patch(rect1)

            """calculate FFT"""
            freq_spec = abs(fft(acoustic_data))
            freq_spec = 20 * np.log10(freq_spec)
            num_datapts = int(np.fix(len(freq_spec) / 2))
            freq_spec = freq_spec[0: num_datapts]
            fft_pts = np.arange(0, num_datapts)
            freq1 = fft_pts * (sampling_rate / (2 * num_datapts))

            """plot signal at frequency domain"""
            plt.subplot(212)
            xtick = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 500000]
            xticklabels = ['0', '50', '100', '150', '200', '250', '300', '350', '500']
            ax1.stem(freq1, freq_spec, markerfmt='None', basefmt='None')
            plt.xticks(xtick, xticklabels)
            # plt.ylim(0, 50)
            plt.xlabel('Frequency(kHz)')
            plt.ylabel('Magnitude(dB)')
            plt.title('Frequency Spectrum')

            fig.tight_layout()
            plt.savefig(os.path.join(figure_path, fig_title + '.png'))
            plt.cla()  # Clear axis
            plt.clf()  # Clear figure
            plt.close()  # Close a figure window

            """Print mel-spectrum figures"""
            fig1, ax = plt.subplots()
            plt.figure(2)
            start_time = filename.split(' ')[1]
            end_time = filename.split(' ')[3].split('-')[0]
            title = 'Defect ' + start_time + ' sec to ' + end_time + ' sec'
            melspec_total = librosa.feature.melspectrogram(acoustic_data, n_mels=512)
            logspec_total = librosa.amplitude_to_db(melspec_total)
            librosa.display.specshow(logspec_total, cmap='jet', sr=sr, x_axis='time', y_axis='linear')
            ax.add_patch(Rectangle((defect_start, 10000), 0.01, 480000, fc='none', ec='r', lw=2, zorder=2))
            plt.colorbar(format='%+2.0f dB')
            plt.title(print_label + ' + ' + defect_type)
            fig1.tight_layout()
            plt.savefig(os.path.join(figure_path, title + 'mel.png'))
            plt.cla()  # Clear axis
            plt.clf()  # Clear figure
            plt.close()  # Close a figure window

        else:
            continue
