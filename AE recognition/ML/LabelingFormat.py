import os
import numpy as np
import pandas as pd
import math
from datetime import datetime
from scipy import stats
from scipy import signal
from decimal import Decimal
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

"""
Defect code: 
Defect = Y
Noise = X
"""

"""Change the parameters below for each test"""
test_num = '4th-'
test_name = test_num + 'test-40mph'
test_day = '1st day'
training_trial_num = '100 kHz'
defect_type = 'copy'
filter_order = 6
filter_frequency = 0.2
fold_name = test_num + test_day
defect_data_path = '/media/lj/Data/Python/HTL/' + test_day + '/labeling-' + test_num + 'test (' + defect_type + ').csv'
sampling_rate = 1000000

"""read csv file and extract data"""
defect_data = pd.read_csv(defect_data_path)
start_time_list = list(defect_data.iloc[:, 2])
end_time_list = list(defect_data.iloc[:, 3])
defect_number = list(defect_data.iloc[:, 1])
# defect_list = list(defect_data.iloc[:, 5])
for i in range(0, len(start_time_list)):
    start_time = start_time_list[i]
    end_time = end_time_list[i]
    label_name = defect_number[i]
    # defect_code = defect_list[i]
    for j in range(1, 4):
        sensor_set = j
        """Fixed parameters"""
        fl_start_point = "{:.3f}".format(start_time % 1)
        start_point = Decimal(fl_start_point)
        fl_end_point = "{:.3f}".format(end_time % 1)
        end_point = Decimal(fl_end_point)
        start_data_number = start_point * sampling_rate
        end_data_number = end_point * sampling_rate
        timestamp = []

        """if time range includes two files, implement the first loop. Otherwise, go to 2nd"""
        if math.floor(start_time) < math.floor(end_time):
            signal_second_1 = math.floor(start_time)
            signal_second_2 = math.floor(end_time)
            file_name_1 = signal_second_1 % 200
            file_name_2 = signal_second_2 % 200
            integer_1 = signal_second_1 // 200 + 1
            integer_2 = signal_second_2 // 200 + 1
            fold_number_1 = '-' + str(integer_1)
            fold_number_2 = '-' + str(integer_2)

            input_path_1 = '/media/lj/Data/Python/HTL/' + test_day + '/' + test_name + '-sensor' + str(
                sensor_set) + '/' + test_name + fold_number_1 + '/' + str(
                file_name_1) + '-' + str(sensor_set) + '.csv'
            input_path_2 = '/media/lj/Data/Python/HTL/' + test_day + '/' + test_name + '-sensor' + str(
                sensor_set) + '/' + test_name + fold_number_2 + '/' + str(
                file_name_2) + '-' + str(sensor_set) + '.csv'
            split_file_1 = pd.read_csv(input_path_1, escapechar="\\")
            split_file_2 = pd.read_csv(input_path_2, escapechar="\\")
            acoustic_data_1 = list(split_file_1.iloc[int(start_data_number):, 1])
            acoustic_data_2 = list(split_file_2.iloc[:int(end_data_number), 1])
            acoustic_data = acoustic_data_1 + acoustic_data_2

            """Scale"""
            acoustic_data_df = pd.DataFrame(acoustic_data)
            acoustic_data_df = acoustic_data_df.apply(lambda x: (x - min(x)) / (max(x) - min(x)), axis=0)
            acoustic_data_scaled = list(acoustic_data_df.iloc[:, 0])
            calibration = np.mean(acoustic_data_scaled)
            acoustic_data_scaled -= calibration
            timelength = len(acoustic_data_scaled)

            """Band-pass filter"""
            b, a = signal.butter(filter_order, filter_frequency, 'highpass')  # setup filter parameters
            acoustic_data_filtered = signal.filtfilt(b, a, acoustic_data_scaled)  # data is signal to be filtered

            # for k in range(0, len(acoustic_data)):
            #     temp = datetime.fromtimestamp(k)
            #     timestamp.append(temp.strftime('%Y-%m-%dT%H:%M:%S.000Z'))

            for k in range(0, timelength):
                timestamp.append(k)

            # split_file_1 = split_file_1.iloc[0:timelength, :]
            # split_file_1.iloc[1:timelength, 0] = 'series_a'
            # split_file_1.iloc[0, 0] = 'series_b'
            # split_file_1.insert(1, 'timestamp', timestamp)
            # split_file_1.iloc[0:timelength, 1] = acoustic_data
            # split_file_1.insert(3, 'label', "")
            # split_file.iloc[0, 3] = "Crack"
            # split_file.iloc[1, 3] = "Wear"
            # split_file_1.iloc[:, 2] = split_file_1.iloc[:, 2].apply(lambda x: '{:.8f}'.format(x))
            # output_file = split_file_1.iloc[0:timelength, :]
            # acoustic_data.reshape(-1, 1)
            # min_max_scaler = preprocessing.MinMaxScaler()
            # acoustic_data_norm = min_max_scaler.fit_transform([acoustic_data])
            ae_dataframe = pd.DataFrame(acoustic_data_filtered)
            # output_file.columns = ['series', 'timestamp', 'value', 'label']
            output_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/' + training_trial_num + \
                          '/Defect/' + defect_type + '/' + test_num + test_day + '/'
            os.makedirs(output_path, exist_ok=True)
            ae_dataframe.to_csv(
                os.path.join(output_path, 'Y ' + str(start_time) + ' sec to ' + str(end_time) + '-' + test_name + '-'
                             + test_day + '@' + str(sensor_set) + '.csv'))

        else:
            signal_second = math.floor(start_time)
            file_name = signal_second % 200
            integer = signal_second // 200 + 1
            fold_number = '-' + str(integer)
            input_path = '/media/lj/Data/Python/HTL/' + test_day + '/' + test_name + '-sensor' + str(sensor_set) + \
                         '/' + test_name + fold_number + \
                         '/' + str(file_name) + '-' + str(sensor_set) + '.csv'
            split_file = pd.read_csv(input_path, escapechar="\\")
            acoustic_data = list(split_file.iloc[int(start_data_number):int(end_data_number), 1])

            """Scale"""
            acoustic_data_df = pd.DataFrame(acoustic_data)
            acoustic_data_df = acoustic_data_df.apply(lambda x: (x - min(x)) / (max(x) - min(x)), axis=0)
            acoustic_data_scaled = list(acoustic_data_df.iloc[:, 0])
            calibration = np.mean(acoustic_data_scaled)
            acoustic_data_scaled -= calibration
            timelength = len(acoustic_data_scaled)

            """Band-pass filter"""
            b, a = signal.butter(filter_order, filter_frequency, 'highpass')  # setup filter parameters
            acoustic_data_filtered = signal.filtfilt(b, a, acoustic_data_scaled)  # data is signal to be filtered

            # for k in range(0, len(acoustic_data)):
            #     temp = datetime.fromtimestamp(k)
            #     timestamp.append(temp.strftime('%Y-%m-%dT%H:%M:%S.000Z'))

            for k in range(0, len(acoustic_data_filtered)):
                timestamp.append(k)

            # split_file = split_file.iloc[0:timelength, :]
            # split_file.iloc[1:timelength, 0] = 'series_a'
            # split_file.iloc[0, 0] = 'series_b'
            # split_file.insert(1, 'timestamp', timestamp)
            # split_file.iloc[0:timelength, 1] = acoustic_data
            # split_file.insert(3, 'label', "")
            # split_file.iloc[0, 3] = "Crack"
            # split_file.iloc[1, 3] = "Wear"
            # split_file.iloc[1:, 2] = split_file.iloc[1:, 2].apply(lambda x: '{:.8f}'.format(x))
            # output_file = split_file.iloc[0:timelength, :]
            # acoustic_data.reshape(-1, 1)
            # min_max_scaler = preprocessing.MinMaxScaler()
            # acoustic_data_norm = min_max_scaler.fit_transform([acoustic_data])
            ae_dataframe = pd.DataFrame(acoustic_data_filtered)
            # output_file.columns = ['series', 'timestamp', 'value', 'label']
            output_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/' + training_trial_num + \
                          '/Defect/' + defect_type + '/' + test_num + test_day + '/'
            os.makedirs(output_path, exist_ok=True)
            ae_dataframe.to_csv(
                os.path.join(output_path, 'Y ' + str(start_time) + ' sec to ' + str(end_time) + '-' + test_name + '-'
                             + test_day + '@' + str(sensor_set) + '.csv'))
