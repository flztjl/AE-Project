import os
import pandas as pd
import numpy as np

file_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/noise/'
# '/media/lj/MachineLearning/AE recognition/Data/HTL data/11th/defect/'

output_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/noise-duplicated/'
# '/media/lj/MachineLearning/AE recognition/Data/HTL data/11th/defect/plot1/'

os.makedirs(output_path, exist_ok=True)
dir_list = os.listdir(file_path)
for filename in dir_list:
    if filename == 'plot':
        continue
    data_type = filename.split(' ')[0]
    split_file = pd.read_csv(os.path.join(file_path, filename))
    acoustic_data = split_file.iloc[:, 1].to_numpy()
    duplicate = acoustic_data[40000: 50000]
    acoustic_data[0: 10000] += duplicate
    scale_rate = 1 / np.sqrt(np.mean(acoustic_data ** 2))
    scaled_acoustic_data = acoustic_data * scale_rate
    acoustic_data_scaled = list(scaled_acoustic_data)
    data_subset = {'sensor': acoustic_data_scaled}
    dataframe = pd.DataFrame(data_subset)
    dataframe.to_csv(os.path.join(output_path, filename))
