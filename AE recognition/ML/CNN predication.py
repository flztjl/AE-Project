import glob
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay

"""Function to extract features and split raw data by time window"""
figure_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/N-old/processed-old//fig/'
os.makedirs(figure_path, exist_ok=True)


def extract_features(parent_dir, sub_dir, file_ext="*.csv",
                     bands=256, frames=98):
    features, labels, file_names = [], [], []
    for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        segment_log_specgrams, segment_labels = [], []
        sound_file = pd.read_csv(fn)
        sound_clip = sound_file.iloc[:, 1]
        sound_clip = sound_clip.to_numpy()
        tag = fn.split('/')[10].split(' ')[0]
        if tag == 'X':
            label = 0
        else:
            label = 1
        file_name = fn.split('/')[10]

        """Calculate Mel-frequency cepstral coefficients (MFCCs)"""
        # for (start, end) in _windows(sound_clip, window_size):
        #     if len(sound_clip[start:end]) == window_size:
        signal = sound_clip  # [start:end]
        melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
        logspec = librosa.amplitude_to_db(melspec)
        logspec = logspec.T.flatten()[:, np.newaxis].T
        segment_log_specgrams.append(logspec)
        segment_labels.append(label)

        """Reshape 1d data into 2d, from 2560 to 64 * 40"""
        segment_log_specgrams = np.asarray(segment_log_specgrams).reshape(
            len(segment_log_specgrams), bands, frames, 1)
        segment_features = np.concatenate((segment_log_specgrams, np.zeros(
            np.shape(segment_log_specgrams))), axis=3)

        """Get the derivative of MFCC as another feature"""
        # [:, :, :, 0] is MFCC, [:, :, :, 1] is the derivative of MFCC
        for i in range(len(segment_features)):
            segment_features[i, :, :, 1] = librosa.feature.delta(
                segment_features[i, :, :, 0])

        if len(segment_features) > 0:
            features.append(segment_features)
            labels.append(segment_labels)
            file_names.append(file_name)
    return features, labels, file_names


"""Pre-process and extract feature from the data"""

parent_dir = '/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/N-old/'
save_dir = '/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/N-old/processed-old/'
os.makedirs(save_dir, exist_ok=True)

folds = sub_dirs = np.array(['1', '2', '3', '4', '5', '6', '7'])

for sub_dir in sub_dirs:
    features, labels, file_names = extract_features(parent_dir, sub_dir)
    np.savez("{0}{1}.npz".format(save_dir, sub_dir),
             features=features, labels=labels, file_names=file_names)

"""Train and evaluate"""

accuracies = []
load_dir = '/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/N-old/processed-old/'

x_test, y_test = [], []

# for testing we will make predictions on each segment and average them to
# produce signle label for an entire sound clip.
test_data = np.load("{0}/{1}.npz".format(load_dir,
                                         folds[0]), allow_pickle=True)
x_test = test_data["features"]
y_test = test_data["labels"]
z_test = test_data["file_names"]

model = keras.models.load_model('/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead '
                                'break/extract/N-0.01dup/processed/model/')
# evaluate on test set/fold
y_true, y_pred, y_name = [], [], []
y_compose = []
for x, y, z in zip(x_test, y_test, z_test):
    # average predictions over segments of a sound clip
    avg_p = np.argmax(np.mean(model.predict(x), axis=0))
    y_pred.append(avg_p)
    # pick single label via np.unique for a sound clip
    y_true.append(y[0])
    y_name.append(z)

# print("Fold Accuracy: {0}".format(y_pred))
# print("Fold Accuracy: {0}".format(y_true))
# print("Fold Accuracy: {0}".format(y))
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
# auc = roc_auc_score(y_true, y_pred)
# print(f"auc={auc}")
print(f"precision={prec}, recall={rec}, f1={f1}")
disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
disp.figure_.savefig(os.path.join(figure_path, str(val_index[0]) + '-cm.png'))
plt.cla()  # Clear axis
plt.clf()  # Clear figure
plt.close()

y_compose = list(zip(y_true, y_pred, y_name))
y_compose_df = pd.DataFrame(y_compose)
y_compose_df.to_csv('/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/N-old/processed/1.csv')
accuracies.append(accuracy_score(y_true, y_pred))
# loss, accuracy, f1_score, precision, recall = model.evaluate(x_test_copy, y_test_copy, verbose=1)
# print("Fold Accuracy: {0}".format(y_pred))
# print("Fold Accuracy: {0}".format(y_compose))
print("Test Accuracy: {0}".format(accuracies))
