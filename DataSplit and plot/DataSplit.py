import pandas as pd
import os
from nptdms import TdmsFile


file_name = 'Void-2nd defect at head-40db'
task_ID = 'Task 1'

rawfile_path = '/Volumes/Extreme SSD/2023-09-08/train day2/' + file_name + '.tdms'

# D:/Google Drive/Colab Notebooks/AE recognition
# /Users/David/Google Drive/Colab Notebooks/AE recognition

output_path_sensor1 = ('/Volumes/Extreme SSD/2023-09-08/train day2/' + file_name + '/')
# output_path_sensor2 = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/plb-museum/1st ' \
#                      'day/' + file_name + '/2/'
# # output_path_sensor3 = '/media/lj/AE Project/Archives/plb-museum/plb field test/4th/' + file_name + '/3/'
# output_path_sensor3 = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/plb-museum/1st ' \
#                       'day/' + file_name + '/3/'

defect_data_path = 'D:/Python/HTL/1st day/labeling-1st test.csv'

# '/media/lj/Data/Python/HTL/1st day/3rd test-20mph-sensor2/3rd test-20mph/'
# /Users/David/Google Drive/Colab Notebooks/AE recognition/Data
# D:/Google Drive/Colab Notebooks/AE recognition/Data/

os.makedirs(output_path_sensor1, exist_ok=True)
# os.makedirs(output_path_sensor2, exist_ok=True)
# os.makedirs(output_path_sensor3, exist_ok=True)


def data_split(filename):
    with TdmsFile.open(filename) as tdms_file:
        channel1 = tdms_file[task_ID]['Untitled']
        # channel2 = tdms_file[task_ID]['Untitled 1']
        # channel3 = tdms_file[task_ID]['Untitled 2']
        # channel4 = tdms_file[task_ID]['Untitled 3']
        # Access dictionary of properties
        # properties = channel.properties
        # Access numpy array of data for channel:
        data1 = channel1[:]
        # data2 = channel2[:]
        # data3 = channel3[:]
        # noinspection PyGlobalUndefined
        # data4 = channel4[:]
        time_line = []
        file_num = int(len(data1) / 1000000)

        # Split raw data into csv files of one second long
        for i in range(0, file_num + 1):
            if i == file_num:
                data_subset1 = {'sensor1': data1[i * 1000000:]}
                # data_subset2 = {'sensor2': data2[i * 1000000:]}
                # data_subset3 = {'sensor3': data3[i * 1000000:]}
                dataframe1 = pd.DataFrame(data_subset1)
                # dataframe2 = pd.DataFrame(data_subset2)
                # dataframe3 = pd.DataFrame(data_subset3)
                dataframe1.to_csv(os.path.join(output_path_sensor1, str(i) + "-1.csv"))
                # dataframe2.to_csv(os.path.join(output_path_sensor2, str(i) + "-2.csv"))
                # dataframe3.to_csv(os.path.join(output_path_sensor3, str(i) + "-3.csv"))
            else:
                data_subset1 = {'sensor1': data1[i * 1000000: (i + 1) * 1000000]}
                # data_subset2 = {'sensor2': data2[i * 1000000: (i + 1) * 1000000]}
                # data_subset3 = {'sensor3': data3[i * 1000000: (i + 1) * 1000000]}
                dataframe1 = pd.DataFrame(data_subset1)
                # dataframe2 = pd.DataFrame(data_subset2)
                # dataframe3 = pd.DataFrame(data_subset3)
                dataframe1.to_csv(os.path.join(output_path_sensor1, str(i) + "-1.csv"))
                # dataframe2.to_csv(os.path.join(output_path_sensor2, str(i) + "-2.csv"))
                # dataframe3.to_csv(os.path.join(output_path_sensor3, str(i) + "-3.csv"))

        # for j in range(0, len(data4), 40000):
        #     time_line.append(data4[j])
        #     dataframe = pd.DataFrame(time_line)
        # dataframe.to_csv(os.path.join(output_path_sensor1, 'timeline.csv'))
