import os
import glob
import shutil
import numpy as np


parent_dir = '/media/lj/MachineLearning/AE recognition/Data/UrbanSound8K/audio/'
output_path = '/media/lj/MachineLearning/AE recognition/Data/UrbanSound8K/newaudio'
os.makedirs(output_path, exist_ok=True)
folds = sub_dirs = np.array(['fold1', 'fold2', 'fold3', 'fold4',
                             'fold5', 'fold6', 'fold7', 'fold8',
                             'fold9', 'fold10'])
for sub_dir in sub_dirs:
    for fn in glob.glob(os.path.join(parent_dir, sub_dir, "*.wav")):
        label = int(fn.split('/')[9].split('-')[1])
        if label == 0 or label == 1:
            shutil.copyfile(fn, os.path.join(output_path, fn.split('/')[9]))
