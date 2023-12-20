"""Load necessary libraries"""
import glob
import os
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow import keras

if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

"""Define helper functions"""


def extract_features(parent_dir, sub_dir, file_ext='*.csv',
                     bands=60, frames=41):
    def _windows(data, window_size):
        start = 0
        while start < len(data):
            yield int(start), int(start + window_size)
            start += (window_size // 2)

    sr = 1000000
    window_size = 512 * (frames - 1)
    features, labels = [], []
    for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        segment_log_specgrams, segment_labels = [], []
        sound_file = pd.read_csv(fn)
        tag = sound_file.bfill(axis=0).iloc[:, 3]
        tag = tag[0]
        if tag == 'Joint':
            label = 0
        else:
            label = 1
        sound_clip = sound_file.iloc[:, 2]
        sound_clip = sound_clip.to_numpy()
        for (start, end) in _windows(sound_clip, window_size):
            if len(sound_clip[start:end]) == window_size:
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                segment_log_specgrams.append(logspec)
                segment_labels.append(label)

        segment_log_specgrams = np.asarray(segment_log_specgrams).reshape(
            len(segment_log_specgrams), bands, frames, 1)
        segment_features = np.concatenate((segment_log_specgrams, np.zeros(
            np.shape(segment_log_specgrams))), axis=3)
        for i in range(len(segment_features)):
            segment_features[i, :, :, 1] = librosa.feature.delta(
                segment_features[i, :, :, 0])

        if len(segment_features) > 0:  # check for empty segments
            features.append(segment_features)
            labels.append(segment_labels)
    return features, labels


"""Pre-process and extract feature from the data"""
parent_dir = '/media/lj/MachineLearning/AE recognition/Data/AE-test10/Labeled data/'
save_dir = "/media/lj/MachineLearning/AE recognition/Data/AE-test10/Labeled data/process/"
folds = sub_dirs = np.array(['fold1', 'fold2', 'fold3', 'fold4',
                             'fold5', 'fold6', 'fold7', 'fold8',
                             'fold9', 'fold10'])
for sub_dir in sub_dirs:
    features, labels = extract_features(parent_dir, sub_dir)
    np.savez("{0}{1}.npz".format(save_dir, sub_dir),
             features=features,
             labels=labels)

"""Define convolutional network architecture"""


def get_network():
    num_filters = [24, 32, 64, 128]
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (60, 41, 2)
    num_classes = 10
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(24, kernel_size,
                                  padding="same", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(32, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(64, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(128, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model


"""Train and evaluate via 10-Folds cross-validation"""
accuracies = []
folds = np.array(['fold1', 'fold2', 'fold3', 'fold4',
                  'fold5', 'fold6', 'fold7', 'fold8',
                  'fold9', 'fold10'])
load_dir = "/media/lj/MachineLearning/AE recognition/Data/AE-test10/Labeled data/process/"

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(folds):
    x_train, y_train = [], []
    for ind in train_index:
        """read features or segments of an audio file"""
        train_data = np.load("{0}{1}.npz".format(load_dir, folds[ind]),
                             allow_pickle=True)
        """for training stack all the segments so that they are treated as an example/instance"""
        features = np.concatenate(train_data["features"], axis=0)
        labels = np.concatenate(train_data["labels"], axis=0)
        x_train.append(features)
        y_train.append(labels)
    # stack x,y pairs of all training folds
    x_train = np.concatenate(x_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.float32)

    # for testing we will make predictions on each segment and average them to
    # produce single label for an entire sound clip.
    test_data = np.load("{0}/{1}.npz".format(load_dir,
                                             folds[test_index][0]), allow_pickle=True)
    x_test = test_data["features"]
    y_test = test_data["labels"]

    soundclassification_model = get_network()
    soundclassification_model.summary()

    """Save checkpoints during training"""
    checkpoint_path = "/media/lj/MachineLearning/AE recognition/Data/model/"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    soundclassification_model.fit(x_train, y_train, epochs=10, batch_size=24, verbose=0, callbacks=[cp_callback])

    """Creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:"""
    os.listdir(checkpoint_dir)

    # evaluate on test set/fold
    y_true, y_pred = [], []
    for x, y in zip(x_test, y_test):
        # average predictions over segments of a sound clip
        avg_p = np.argmax(np.mean(soundclassification_model.predict(x), axis=0))
        y_pred.append(avg_p)
        # pick single label via np.unique for a sound clip
        y_true.append(np.unique(y)[0])
    accuracies.append(accuracy_score(y_true, y_pred))
    print("10 Folds Accuracy: {0}".format(np.mean(accuracies)))

