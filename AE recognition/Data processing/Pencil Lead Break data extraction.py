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

"""Change the parameters below for each test"""
test_number = 'test7'
filter_order = 3
filter_frequency = 0.06
defect_data_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/' + test_number + '.csv'
sampling_rate = 1000000

"""read csv file and extract data"""
defect_data = pd.read_csv(defect_data_path)
start_time_list = list(defect_data.iloc[:, 1])
end_time_list = list(defect_data.iloc[:, 2])
for i in range(0, len(start_time_list)):
    start_time = start_time_list[i]
    end_time = end_time_list[i]
    file_name = math.floor(start_time)
    fl_start_point = "{:.2f}".format(start_time - file_name)
    start_point = Decimal(fl_start_point)
    fl_end_point = "{:.2f}".format(end_time - file_name)
    end_point = Decimal(fl_end_point)
    start_data_number = start_point * sampling_rate
    end_data_number = end_point * sampling_rate

    input_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/' + test_number\
                 + '/' + str(file_name) + '.csv'
    split_file = pd.read_csv(input_path, escapechar="\\")
    acoustic_data = list(split_file.iloc[int(start_data_number):int(end_data_number), 1])

    """Scale"""
    acoustic_data_ar = np.asarray(acoustic_data)
    scale_rate = 1 / np.sqrt(np.mean(acoustic_data_ar ** 2))
    # acoustic_data_df = pd.DataFrame(acoustic_data)
    # acoustic_data_df = acoustic_data_df.apply(lambda x: 2 * (x - min(x)) / (max(x) - min(x)), axis=0)
    scaled_acoustic_data = acoustic_data_ar * scale_rate
    acoustic_data_scaled = list(scaled_acoustic_data)
    # acoustic_data_scaled = list(acoustic_data_df.iloc[:, 0])
    calibration = np.mean(acoustic_data_scaled)
    acoustic_data_scaled -= calibration
    timelength = len(acoustic_data_scaled)

    """Band-pass filter"""
    b, a = signal.butter(filter_order, filter_frequency, 'highpass')  # setup filter parameters
    acoustic_data_filtered = signal.filtfilt(b, a, acoustic_data_scaled)  # data is signal to be filtered

    ae_dataframe = pd.DataFrame(acoustic_data_filtered)
    output_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/'\
                  + test_number + '/'
    os.makedirs(output_path, exist_ok=True)
    ae_dataframe.to_csv(
        os.path.join(output_path, str(start_time) + ' sec to ' + str(end_time) + '.csv'))
