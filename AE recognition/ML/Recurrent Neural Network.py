"""Load necessary libraries"""
import glob
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score

"""Define helper functions"""


def extract_features(parent_dir, sub_dirs, file_ext="*.csv",
                     bands=172, frames=160):
    # def _windows(data, window_size):
    #     start = 0
    #     while start < len(data):
    #         yield start, start + window_size
    #         start += (window_size // 2)

    # window_size = 512 * (frames - 1)
    features, labels = [], []
    for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        segment_mfcc, segment_labels = [], []
        sound_file = pd.read_csv(fn)
        sound_clip = sound_file.iloc[:, 1]
        sound_clip = sound_clip.to_numpy()
        tag = fn.split('/')[9].split(' ')[0]
        if tag == 'X':
            label = 0
        else:
            label = 1

        signal = sound_clip
        mfcc = librosa.feature.mfcc(y=signal, sr=100000,
                                    n_mfcc=bands).T.flatten()[:, np.newaxis].T
        segment_mfcc.append(mfcc)
        segment_labels.append(label)

        segment_mfcc = np.asarray(segment_mfcc).reshape(
            len(segment_mfcc), frames, bands)

        if len(segment_mfcc) > 0:  # check for empty segments
            features.append(segment_mfcc)
            labels.append(segment_labels)

    return features, labels


parent_dir = '/media/lj/MachineLearning/AE recognition/Data/HTL data/6th/'
save_dir = '/media/lj/MachineLearning/AE recognition/Data/HTL data/6th/RNN/processed/'
os.makedirs(save_dir, exist_ok=True)

folds = sub_dirs = np.array(['training fold', 'testing fold', 'validation fold'])

for sub_dir in sub_dirs:
    features, labels = extract_features(parent_dir, sub_dir)
    np.savez("{0}{1}".format(save_dir, sub_dir), features=features,
             labels=labels)

"""Define GPU based recurrent network architecture"""


def get_network():
    input_shape = (160, 172)
    num_classes = 2
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(128, input_shape=input_shape))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    return model


"""Train and evaluate via 10-Folds cross-validation"""
accuracies = []

load_dir = '/media/lj/MachineLearning/AE recognition/Data/HTL data/6th/RNN/processed/'

# read features or segments of an audio file
train_data = np.load("{0}/{1}.npz".format(load_dir, 'training fold'),
                     allow_pickle=True)
# for training stack all the segments so that they are treated as an example/instance
x_train = train_data["features"]
y_train = train_data["labels"]
# stack x,y pairs of all training folds
x_train = np.concatenate(x_train, axis=0).astype(np.float32)
y_train = np.concatenate(y_train, axis=0).astype(np.float32)


validation_data = np.load("{0}/{1}.npz".format(load_dir, 'validation fold'), allow_pickle=True)
x_validation = validation_data["features"]
y_validation = validation_data["labels"]
y_validation = np.reshape(y_validation, (-1, len(y_validation)))
x_validation = np.concatenate(x_validation, axis=0).astype(np.float32)
y_validation = np.concatenate(y_validation, axis=0).astype(np.float32)
# for testing we will make predictions on each segment and average them to
# produce signle label for an entire sound clip.
test_data = np.load("{0}/{1}.npz".format(load_dir, 'testing fold'), allow_pickle=True)
x_test = test_data["features"]
y_test = test_data["labels"]

model = get_network()
model.summary()

checkpoint_dir = "/media/lj/MachineLearning/AE recognition/Data/HTL data/6th/model/RNN/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                 save_weights_only=True,
                                                 verbose=1)
history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20, batch_size=24,
                    verbose=1, callbacks=[cp_callback])

os.listdir(checkpoint_dir)
print(history.history.keys())

plt.plot(history.history['accuracy'], 'r')
plt.plot(history.history['loss'], 'b')
plt.plot(history.history['val_accuracy'], 'g')
plt.plot(history.history['val_loss'], 'm')
plt.title('model accuracy & loss')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss', 'val_accuracy', 'val_loss'], loc='upper left')
plt.show()

# evaluate on test set/fold
y_true, y_pred = [], []
for x, y in zip(x_test, y_test):
    # average predictions over segments of a sound clip
    avg_p = np.argmax(np.mean(model.predict(x), axis=0))
    y_pred.append(avg_p)
    # pick single label via np.unique for a sound clip
    y_true.append(y[0])
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
auc = roc_auc_score(y_true, y_pred)
print(f"auc={auc}")
print(f"precision={prec}, recall={rec}, f1={f1}")
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.show()
accuracies.append(accuracy_score(y_true, y_pred))
print("Average Accuracy: {0}".format(accuracies))
