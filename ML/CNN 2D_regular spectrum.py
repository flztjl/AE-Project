### Load necessary libraries ###
import glob
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import tensorflow as tf
from sklearn.model_selection import KFold
from scipy import signal
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay
from tensorflow import keras
from sklearn.metrics import roc_auc_score

"""Confirm GPU is computing"""
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

"""Clean GPU memory"""
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""Fixed parameters: sr = sampling rate, figure_path refers to plotting function in test section"""
sr = 1000000
figure_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead ' \
              'break/extract/N/fig/'
os.makedirs(figure_path, exist_ok=True)

"""Function to extract features and split raw data by time window"""


def extract_features(parent_dir, sub_dir, file_ext="*.csv",
                     bands=223, frames=129):
    # def _windows(data, window_size):
    #     start = 0
    #     while start < len(data):
    #         yield int(start), int(start + window_size)
    #         start += (window_size // 2)
    #
    # window_size = 500 * frames
    features, labels, file_names = [], [], []

    """read each file in the folder and extract and label"""
    for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        segment_log_specgrams, segment_labels = [], []
        sound_file = pd.read_csv(fn)
        sound_clip = sound_file.iloc[:, 1]
        sound_clip = sound_clip.to_numpy()
        tag = fn.split('/')[11].split(' ')[0]
        if tag == 'X':
            label = 0
        else:
            label = 1
        file_name = fn.split('/')[11].split('.csv')[0]

        """Print mel-spectrum figures"""
        # start_time = fn.split('/')[8].split(' ')[1]
        # end_time = fn.split('/')[8].split(' ')[4]
        # fig = plt.figure(figsize=(12, 6))
        # title = tag + ' ' + start_time + ' sec to ' + end_time + ' sec'
        # melspec_total = librosa.feature.melspectrogram(sound_clip, n_mels=bands)
        # logspec_total = librosa.amplitude_to_db(melspec_total)
        # librosa.display.specshow(logspec_total, sr=sr, x_axis='time', y_axis='linear')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Spectrogram')
        # fig.tight_layout()
        # plt.savefig(os.path.join(figure_path, title + '.png'))
        # plt.cla()  # Clear axis
        # plt.clf()  # Clear figure
        # plt.close()  # Close a figure window

        """Calculate Mel-frequency cepstral coefficients (MFCCs)"""
        # for (start, end) in _windows(sound_clip, window_size):
        #     if len(sound_clip[start:end]) == window_size:
        # signal = sound_clip  # [start:end]
        _, _, sgram = signal.spectrogram(sound_clip, sr, nperseg=256, scaling='spectrum')
        logspec_total = librosa.amplitude_to_db(sgram * 10000)
        # melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
        logspec = librosa.amplitude_to_db(logspec_total)
        logspec = logspec.T.flatten()[:, np.newaxis].T
        segment_log_specgrams.append(logspec)
        # dec_lv = 8
        # data_matrix = []
        # data_dec = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=dec_lv)
        # for j in [node.path for node in data_dec.get_level(dec_lv, 'freq')]:
        #     data_array = data_dec[j].data
        #     data_matrix.append(data_array)
        # data_matrix = np.asarray(data_matrix).flatten()[:, np.newaxis].T
        # segment_log_specgrams.append(data_matrix)

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

        # segment_features = segment_log_specgrams

        # check for empty segments
        if len(segment_features) > 0:
            features.append(segment_features)
            labels.append(segment_labels)
            file_names.append(file_name)
    return features, labels, file_names


"""Pre-process and extract feature from the data"""

parent_dir = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/N/'
save_dir = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/N/processed/'
os.makedirs(save_dir, exist_ok=True)

folds = sub_dirs = np.array(['1', '2', '3', '4', '5', '6'])
folds_model = np.array(['1', '2', '3', '4', '5'])

for sub_dir in sub_dirs:
    features, labels, file_names = extract_features(parent_dir, sub_dir)
    np.savez("{0}{1}.npz".format(save_dir, sub_dir),
             features=features, labels=labels, file_names=file_names)

"""Design convolutional network architecture"""


def get_network():
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (223, 129, 2)
    num_classes = 2
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(16, kernel_size,
                                  padding="same", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(16, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(32, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(32, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model


"""Train and evaluate"""

accuracies = []
load_dir = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/N/processed/'
kf = KFold(n_splits=5)
for train_index, val_index in kf.split(folds_model):
    x_train, y_train = [], []
    x_test, y_test = [], []
    for ind in train_index:
        # read features or segments of an audio file
        train_data = np.load("{0}/{1}.npz".format(load_dir, folds_model[ind]),
                             allow_pickle=True)
        # for training stack all the segments so that they are treated as an example/instance
        # x_train = train_data["features"]
        # y_train = train_data["labels"]
        # y_train = np.reshape(y_train, (-1, len(y_train)))
        features = np.concatenate(train_data["features"], axis=0)
        labels = np.concatenate(train_data["labels"], axis=0)
        x_train.append(features)
        y_train.append(labels)
    x_train = np.concatenate(x_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.float32)

    # for testing we will make predictions on each segment and average them to
    # produce signle label for an entire sound clip.
    test_data = np.load("{0}/{1}.npz".format(load_dir, '6'), allow_pickle=True)
    x_test = test_data["features"]
    y_test = test_data["labels"]
    z_test = test_data["file_names"]

    validation_data = np.load("{0}/{1}.npz".format(load_dir, folds_model[val_index][0]), allow_pickle=True)
    x_val = validation_data["features"]
    y_val = validation_data["labels"]

    y_validation = np.reshape(y_test, (-1, len(y_val)))
    x_validation = np.concatenate(x_val, axis=0).astype(np.float32)
    y_validation = np.concatenate(y_val, axis=0).astype(np.float32)

    model = get_network()
    model.summary()

    """Save checkpoints during training"""

    checkpoint_dir = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead ' \
                     'break/extract/N/processed/model/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                     save_weights_only=True,
                                                     verbose=1)
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20, batch_size=24,
                        verbose=1, callbacks=[cp_callback])

    print(str(val_index[0]) + 'th FOLD')

    model.save('/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead '
               'break/extract/N/processed/model/')
    """Creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:"""
    os.listdir(checkpoint_dir)
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history.history['accuracy'], 'r')
    plt.plot(history.history['loss'], 'b')
    plt.plot(history.history['val_accuracy'], 'g')
    plt.plot(history.history['val_loss'], 'm')
    plt.title('model accuracy & loss')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss', 'val_accuracy', 'val_loss'], loc='upper left')
    plt.savefig(os.path.join(figure_path, str(val_index[0]) + '.png'))
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure
    plt.close()  # Close a figure window
    # plt.show()

    """evaluate on test set/fold"""

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
    y_compose_df.to_csv('/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead '
                        'break/extract/N/processed/' + str(val_index[0]) + '.csv')
    accuracies.append(accuracy_score(y_true, y_pred))
# loss, accuracy, f1_score, precision, recall = model.evaluate(x_test_copy, y_test_copy, verbose=1)
# print("Fold Accuracy: {0}".format(y_pred))
# print("Fold Accuracy: {0}".format(y_compose))
print("Test Accuracy: {0}".format(accuracies))
