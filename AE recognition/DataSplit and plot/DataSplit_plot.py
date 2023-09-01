import pandas as pd
import os
from nptdms import TdmsFile


file_name = 'test1-wet surface'
task_ID = 'Task 1'

rawfile_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/test tdms/' + file_name + '.tdms'
# D:/Google Drive/Colab Notebooks/AE recognition
# /Users/David/Google Drive/Colab Notebooks/AE recognition

output_path_sensor = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/test-wet/' + file_name + '/'
# output_path_sensor2 = 'D:/Python/HTL/1st day/7th test-40mph-sensor2/7th test-40mph-2/'
# # '/media/lj/Data/Python/HTL/1st day/3rd test-20mph-sensor2/3rd test-20mph/'
# output_path_sensor3 = 'D:/Python/HTL/1st day/4th test-40mph-sensor3/4th test-40mph-2'

defect_data_path = 'D:/Python/HTL/1st day/labeling-1st test.csv'

# /Users/David/Google Drive/Colab Notebooks/AE recognition/Data
# D:/Google Drive/Colab Notebooks/AE recognition/Data/

os.makedirs(output_path_sensor, exist_ok=True)
# os.makedirs(output_path_sensor2, exist_ok=True)
# os.makedirs(output_path_sensor3, exist_ok=True)


def data_split(filename):
    with TdmsFile.open(filename) as tdms_file:
        channel1 = tdms_file[task_ID]['Untitled']
        # channel2 = tdms_file[task_ID]['Untitled 1']
        # channel3 = tdms_file[task_ID]['Untitled 2']
        # channel4 = tdms_file[task_ID]['Untitled 3']
        # Access dictionary  of properties
        # properties = channel.properties
        # Access numpy array of data for channel:
        data1 = channel1[:]
        # data2 = channel2[:]
        # data3 = channel3[:]
        # noinspection PyGlobalUndefined
        num_files = len(channel1[:]) // 1000000
        # data4 = channel4[:]
        time_line = []

        # Split raw data into csv files of one second long
        for i in range(0, num_files + 1):
            if i == num_files:
                data_subset1 = {'sensor1': data1[i * 1000000:]}
                # data_subset2 = {'sensor2': data2[i * 1000000:]}
                # data_subset3 = {'sensor3': data3[i * 1000000:]}
                dataframe1 = pd.DataFrame(data_subset1)
                # dataframe2 = pd.DataFrame(data_subset2)
                # dataframe3 = pd.DataFrame(data_subset3)
                dataframe1.to_csv(os.path.join(output_path_sensor, str(i) + ".csv"))
                # dataframe2.to_csv(os.path.join(output_path_sensor, str(i) + "-2.csv"))
                # dataframe3.to_csv(os.path.join(output_path_sensor, str(i) + "-3.csv"))
            else:
                data_subset1 = {'sensor1': data1[i * 1000000: (i + 1) * 1000000]}
                # data_subset2 = {'sensor2': data2[i * 1000000: (i + 1) * 1000000]}
                # data_subset3 = {'sensor3': data3[i * 1000000: (i + 1) * 1000000]}
                dataframe1 = pd.DataFrame(data_subset1)
                # dataframe2 = pd.DataFrame(data_subset2)
                # dataframe3 = pd.DataFrame(data_subset3)
                dataframe1.to_csv(os.path.join(output_path_sensor, str(i) + ".csv"))
                # dataframe2.to_csv(os.path.join(output_path_sensor, str(i) + "-2.csv"))
                # dataframe3.to_csv(os.path.join(output_path_sensor, str(i) + "-3.csv"))

        # for j in range(0, len(data4), 40000):
        #     time_line.append(data4[j])
        #     dataframe = pd.DataFrame(time_line)
        # dataframe.to_csv(os.path.join(output_path_sensor1, 'timeline.csv'))
        # return num_files
