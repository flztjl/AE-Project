### Load necessary libraries ###
import glob
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tensorflow import keras

"""Confirm GPU is computing"""
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

"""Clean GPU memory"""
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""Fixed parameters"""
sr = 44100
figure_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/mel figures/'


"""Function to extract features and split raw data by time window"""


def extract_features(parent_dir, sub_dir, file_ext="*.wav",
                     bands=64, frames=40):
    def _windows(data, window_size):
        start = 0
        while start < len(data):
            yield int(start), int(start + window_size)
            start += (window_size // 2)

    window_size = 500 * frames
    features, labels = [], []
    for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        segment_log_specgrams, segment_labels = [], []
        sound_clip, sr = librosa.load(fn)
        label = int(fn.split('/')[8].split('-')[1])


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
        for (start, end) in _windows(sound_clip, window_size):
            if len(sound_clip[start:end]) == window_size:
                signal = sound_clip[start:end]
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

        # check for empty segments
        if len(segment_features) > 0:
            features.append(segment_features)
            labels.append(segment_labels)
    return features, labels


"""Pre-process and extract feature from the data"""

parent_dir = '/media/lj/AE Project/Archives/UrbanSound8K/newaudio/'
save_dir = '/media/lj/AE Project/Archives/UrbanSound8K/newaudio/processed/'

folds = sub_dirs = np.array(['train', 'validate', 'test'])
# for sub_dir in sub_dirs:
#     features, labels = extract_features(parent_dir, sub_dir)
#     np.savez("{0}{1}.npz".format(save_dir, sub_dir),
#              features=features,
#              labels=labels)


"""Design convolutional network architecture"""


def get_network():
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (64, 40, 2)
    num_classes = 2
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(8, kernel_size,
                                  padding="same", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(8, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(8, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(16, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model


"""Train and evaluate"""

accuracies = []
# folds = np.array(['fold1', 'fold2', 'fold3', 'fold4','fold5'])
load_dir = '/media/lj/AE Project/Archives/UrbanSound8K/newaudio/processed/'
# kf = KFold(n_splits=5)
# for train_index, test_index in kf.split(folds):
# for ind in train_index:
# read features or segments of an audio file
train_data = np.load("{0}/{1}.npz".format(load_dir, 'train'), allow_pickle=True)
# for training stack all the segments so that they are treated as an example/instance
x_train = train_data["features"]
y_train = train_data["labels"]
x_train = np.concatenate(x_train, axis=0).astype(np.float32)
y_train = np.concatenate(y_train, axis=0).astype(np.float32)

# for testing we will make predictions on each segment and average them to
# produce signle label for an entire sound clip.

validation_data = np.load("{0}/{1}.npz".format(load_dir, 'validate'), allow_pickle=True)
x_validation = validation_data["features"]
y_validation = validation_data["labels"]
# y_validation = np.reshape(y_validation, (-1, len(y_validation)))
x_validation = np.concatenate(x_validation, axis=0).astype(np.float32)
y_validation = np.concatenate(y_validation, axis=0).astype(np.float32)


test_data = np.load("{0}/{1}.npz".format(load_dir, 'test'), allow_pickle=True)
x_test = test_data["features"]
y_test = test_data["labels"]


model = get_network()
model.summary()

"""Save checkpoints during training"""

checkpoint_path = "/media/lj/MachineLearning/AE recognition/Data/model/"
os.makedirs(checkpoint_path, exist_ok=True)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=30, batch_size=24,
                    verbose=1, callbacks=[cp_callback])

"""Creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:"""
os.listdir(checkpoint_path)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'], 'r')
plt.plot(history.history['loss'], 'b')
plt.plot(history.history['val_accuracy'], 'g')
plt.plot(history.history['val_loss'], 'm')
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy & loss')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss', 'val_accuracy', 'val_loss'], loc='upper left')
plt.show()


"""evaluate on test set/fold"""

y_true, y_pred = [], []
for x, y in zip(x_test, y_test):
    # average predictions over segments of a sound clip
    avg_p = np.argmax(np.mean(model.predict(x), axis=0))
    y_pred.append(avg_p)
    # pick single label via np.unique for a sound clip
    y_true.append(np.unique(y)[0])

    # print("Fold Accuracy: {0}".format(y_pred))
    # print("Fold Accuracy: {0}".format(y_true))
    # print("Fold Accuracy: {0}".format(y))
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
auc = roc_auc_score(y_true, y_pred)
print(f"auc={auc}")
print(f"precision={prec}, recall={rec}, f1={f1}")

accuracies.append(accuracy_score(y_true, y_pred))
# print("Fold Accuracy: {0}".format(y_pred))
# print("Fold Accuracy: {0}".format(y_compose))
print("Test Accuracy: {0}".format(accuracies))
