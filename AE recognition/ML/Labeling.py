import os
import numpy as np
import pandas as pd
import math
from datetime import datetime
from scipy import signal
from decimal import Decimal


def auto_labeling(defect_data_path):
    """The only argument need to change is the defect_data_path in DataSplit"""
    sampling_rate = 1000000
    defect_data = pd.read_csv(defect_data_path)
    start_time_list = list(defect_data.iloc[1:, 2])
    end_time_list = list(defect_data.iloc[1:, 3])
    for i in range(0, len(start_time_list)):
        start_time = start_time_list[i]
        end_time = end_time_list[i]
        for j in range(1, 4):
            sensor_set = j
            """Fixed parameters"""
            fl_start_point = "{:.2f}".format(start_time % 1)
            start_point = Decimal(fl_start_point)
            fl_end_point = "{:.2f}".format(end_time % 1)
            end_point = Decimal(fl_end_point)
            start_data_number = start_point * sampling_rate
            end_data_number = end_point * sampling_rate
            label_name = "{:.2f}".format(start_time) + ' sec to ' + "{:.2f}".format(end_time) + ' sec'
            timestamp = []
            if math.floor(start_time) < math.floor(end_time):
                signal_second_1 = math.floor(start_time)
                signal_second_2 = math.floor(end_time)
                file_name_1 = signal_second_1 % 200
                file_name_2 = signal_second_2 % 200
                integer_1 = signal_second_1 // 200 + 1
                integer_2 = signal_second_2 // 200 + 1
                if integer_1 == 1:
                    fold_number_1 = ''
                else:
                    fold_number_1 = '-' + str(integer_1)

                if integer_2 == 1:
                    fold_number_2 = ''
                else:
                    fold_number_2 = '-' + str(integer_2)

                input_path_1 = 'D:/Python/HTL/1st day/2nd test-20mph-sensor' + str(
                    sensor_set) + '/2nd test-20mph' + fold_number_1 + '/' + str(
                    file_name_1) + '-' + str(sensor_set) + '.csv'
                input_path_2 = 'D:/Python/HTL/1st day/2nd test-20mph-sensor' + str(
                    sensor_set) + '/2nd test-20mph' + fold_number_2 + '/' + str(
                    file_name_2) + '-' + str(sensor_set) + '.csv'
                split_file_1 = pd.read_csv(input_path_1, escapechar="\\")
                split_file_2 = pd.read_csv(input_path_2, escapechar="\\")
                acoustic_data_1 = list(split_file_1.iloc[int(start_data_number):, 1])
                acoustic_data_2 = list(split_file_2.iloc[:int(end_data_number), 1])
                acoustic_data = acoustic_data_1 + acoustic_data_2
                # acoustic_data = np.array(acoustic_data)
                # acoustic_data = acoustic_data.flatten()
                timelength = len(acoustic_data)

                """Band-pass filter"""
                b, a = signal.butter(10, 0.12, 'highpass')  # setup filter parameters
                acoustic_data = signal.filtfilt(b, a, acoustic_data)  # data is signal to be filtered

                # for k in range(0, len(acoustic_data)):
                #     temp = datetime.fromtimestamp(k)
                #     timestamp.append(temp.strftime('%Y-%m-%dT%H:%M:%S.000Z'))

                for k in range(0, 50000):
                    timestamp[k] = k

                split_file_1 = split_file_1.iloc[0:timelength, :]
                split_file_1.iloc[1:timelength, 0] = 'series_a'
                split_file_1.iloc[0, 0] = 'series_b'
                split_file_1.insert(1, 'timestamp', timestamp)
                split_file_1.iloc[0:timelength, 2] = acoustic_data
                split_file_1.insert(3, 'label', "")
                # split_file.iloc[0, 3] = "Crack"
                # split_file.iloc[1, 3] = "Wear"
                split_file_1.iloc[:, 2] = split_file_1.iloc[:, 2].apply(lambda x: '{:.8f}'.format(x))
                output_file = split_file_1.iloc[0:timelength, :]
                output_file.columns = ['series', 'timestamp', 'value', 'label']
                output_path = 'D:/Python/HTL/1st day/2nd test-20mph-sensor' + str(sensor_set) + '/'\
                              + 'to be labeled/'
                os.makedirs(output_path, exist_ok=True)
                output_file.to_csv(os.path.join(output_path, label_name + '-' + str(sensor_set) + '.csv'), index=False)

            else:
                signal_second = math.floor(start_time)
                file_name = signal_second % 200
                integer = signal_second // 200 + 1
                if integer == 1:
                    fold_number = ''
                else:
                    fold_number = '-' + str(integer)
                input_path = 'D:/Python/HTL/1st day/2nd test-20mph-sensor' + str(sensor_set) + \
                             '/2nd test-20mph' + fold_number + \
                             '/' + str(file_name) + '-' + str(sensor_set) + '.csv'
                split_file = pd.read_csv(input_path, escapechar="\\")
                acoustic_data = list(split_file.iloc[int(start_data_number):int(end_data_number), 1])
                timelength = len(acoustic_data)

                """Band-pass filter"""
                b, a = signal.butter(10, 0.12, 'highpass')  # setup filter parameters
                acoustic_data = signal.filtfilt(b, a, acoustic_data)  # data is signal to be filtered

                for k in range(0, len(acoustic_data)):
                    temp = datetime.fromtimestamp(k)
                    timestamp.append(temp.strftime('%Y-%m-%dT%H:%M:%S.000Z'))

                split_file = split_file.iloc[0:timelength, :]
                split_file.iloc[1:timelength, 0] = 'series_a'
                split_file.iloc[0, 0] = 'series_b'
                split_file.insert(1, 'timestamp', timestamp)
                split_file.iloc[0:timelength, 2] = acoustic_data
                split_file.insert(3, 'label', "")
                # split_file.iloc[0, 3] = "Crack"
                # split_file.iloc[1, 3] = "Wear"
                split_file.iloc[1:, 2] = split_file.iloc[1:, 2].apply(lambda x: '{:.8f}'.format(x))
                output_file = split_file.iloc[0:timelength, :]
                output_file.columns = ['series', 'timestamp', 'value', 'label']
                output_path = 'D:/Python/HTL/1st day/2nd test-20mph-sensor' + str(sensor_set) + \
                              '/' + 'to be labeled/'
                os.makedirs(output_path, exist_ok=True)
                output_file.to_csv(os.path.join(output_path, label_name + '-' + str(sensor_set) + '.csv'), index=False)
