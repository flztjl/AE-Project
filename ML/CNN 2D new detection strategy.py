### Load necessary libraries ###
import glob
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
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
figure_path = '/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/processed/figures/'
os.makedirs(figure_path, exist_ok=True)

"""Function to extract features and split raw data by time window"""


def extract_features(parent_dir, sub_dir, file_ext="*.csv",
                     bands=256, frames=215):
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
        tag = fn.split('/')[10].split(' ')[0]
        if tag == 'X':
            label = 0
        else:
            label = 1
        file_name = fn.split('/')[10]

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

        # check for empty segments
        if len(segment_features) > 0:
            features.append(segment_features)
            labels.append(segment_labels)
            file_names.append(file_name)
    return features, labels, file_names


"""Pre-process and extract feature from the data"""

parent_dir = '/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/N/'
save_dir = '/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/processed/'
os.makedirs(save_dir, exist_ok=True)

folds = sub_dirs = np.array(['1', '2', '3', '4', '5', '6'])

# for sub_dir in sub_dirs:
#     features, labels, file_names = extract_features(parent_dir, sub_dir)
#     np.savez("{0}{1}.npz".format(save_dir, sub_dir),
#              features=features, labels=labels, file_names=file_names)

"""Design convolutional network architecture"""


def get_network():
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (256, 215, 2)
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

    model.add(keras.layers.Conv2D(16, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    #
    # model.add(keras.layers.Conv2D(64, kernel_size,
    #                               padding="same"))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model


"""Train and evaluate"""

accuracies = []
load_dir = '/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/processed/'
kf = KFold(n_splits=6)
for train_index, test_index in kf.split(folds):
    x_train, y_train = [], []
    x_test, y_test = [], []
    print(str(test_index[0]) + 'th FOLD')
    for ind in train_index:
        # read features or segments of an audio file
        train_data = np.load("{0}/{1}.npz".format(load_dir, folds[ind]),
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
    test_data = np.load("{0}/{1}.npz".format(load_dir,
                                             folds[test_index][0]), allow_pickle=True)
    x_test = test_data["features"]
    y_test = test_data["labels"]
    z_test = test_data["file_names"]

    # validation_data = np.load("{0}/{1}.npz".format(load_dir, 'validation fold'), allow_pickle=True)
    # x_validation = validation_data["features"]
    # y_validation = validation_data["labels"]
    y_validation = np.reshape(y_test, (-1, len(y_test)))
    x_validation = np.concatenate(x_test, axis=0).astype(np.float32)
    y_validation = np.concatenate(y_validation, axis=0).astype(np.float32)

    model = get_network()
    model.summary()

    """Save checkpoints during training"""

    checkpoint_dir = "/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/model/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                     save_weights_only=True,
                                                     verbose=1)
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=30, batch_size=24,
                        verbose=1, callbacks=[cp_callback])

    model.save('/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/model/')
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
    plt.savefig(os.path.join(figure_path, str(test_index[0]) + '.png'))
    plt.show()

    """evaluate on test set/fold"""

    y_true, y_pred, y_name = [], [], []
    y_compose = []
    y_decision = []
    for x, y, z in zip(x_test, y_test, z_test):
        # average predictions over segments of a sound clip
        avg_p = np.argmax(np.mean(model.predict(x), axis=0))
        y_pred.append(avg_p)
        # pick single label via np.unique for a sound clip
        y_true.append(y[0])
        y_name.append(z)

    y_compose = list(zip(y_name, y_true, y_pred))
    y_compose_df = pd.DataFrame(y_compose)
    index_new_df = int(len(y_true) / 3)
    y_compose_df_new = pd.DataFrame(columns=['defect', 'sensor1', 'sensor2', 'sensor3', 'decision', 'true'],
                                    index=range(index_new_df))

    data_index = 0
    for result_index in range(0, len(y_true)):
        file_name = y_compose_df.iloc[result_index, 0].split('@')[0]
        sensor = y_compose_df.iloc[result_index, 0].split('@')[1].split('.')[0]
        true_label = y_compose_df.iloc[result_index, 1]
        pred_label = y_compose_df.iloc[result_index, 2]
        if file_name not in y_compose_df_new.values[:, 0]:
            y_compose_df_new.iloc[data_index, 0] = file_name
            y_compose_df_new.iloc[data_index, int(sensor)] = pred_label
            y_compose_df_new.iloc[data_index, 5] = true_label
            data_index += 1
        if file_name in y_compose_df_new.values[:, 0]:
            data_index_exist = list(y_compose_df_new.iloc[:, 0]).index(file_name)
            y_compose_df_new.iloc[data_index_exist, int(sensor)] = pred_label

    for result_index in range(0, len(y_compose_df_new.iloc[:, 0])):
        decision_weight = int(y_compose_df_new.iloc[result_index, 1]) + int(y_compose_df_new.iloc[result_index, 2]) \
                          + int(y_compose_df_new.iloc[result_index, 3])
        if decision_weight >= 1:
            y_compose_df_new.iloc[result_index, 4] = 1
        else:
            y_compose_df_new.iloc[result_index, 4] = 0

    y_true = list(y_compose_df_new.iloc[:, 5])
    y_pred = list(y_compose_df_new.iloc[:, 4])
    y_name = list(y_compose_df_new.iloc[:, 0])

    # print("Fold Accuracy: {0}".format(y_pred))
    # print("Fold Accuracy: {0}".format(y_true))
    # print("Fold Accuracy: {0}".format(y))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    # auc = roc_auc_score(y_true, y_pred)
    # print(f"auc={auc}")
    print(f"precision={prec}, recall={rec}, f1={f1}")
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()

    y_compose_df_final = pd.DataFrame((zip(y_name, y_true, y_pred)))

    y_compose_df_final.to_csv('/media/lj/MachineLearning/AE recognition/Data/HTL data/4th/processed/'
                              + str(test_index[0] + 1) + '.csv')
    accuracies.append(accuracy_score(y_true, y_pred))
# loss, accuracy, f1_score, precision, recall = model.evaluate(x_test_copy, y_test_copy, verbose=1)
# print("Fold Accuracy: {0}".format(y_pred))
# print("Fold Accuracy: {0}".format(y_compose))
print("Test Accuracy: {0}".format(accuracies))
