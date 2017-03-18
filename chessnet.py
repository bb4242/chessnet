#!/usr/bin/python

from datetime import datetime
import sys
import os
from glob import glob

import numpy as np
import git

from keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout, Lambda, merge, Merge, Embedding
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K

from preprocessor import Preprocessor


def rel_acc(y_true, y_pred):
    """Relative accuracy metric that calculates the percentage of times
    that the correct move is chosen over the random move. Relies on the
    correct and random moves being stored sequentially in the input and
    output tensors of the network.
    """
    y_neg = y_pred * (2*y_true - 1)
    y_pairs = K.reshape(y_neg, (-1, 2))
    y_sum = K.sum(y_pairs, axis=-1)
    return K.mean(K.greater(y_sum, 0), axis=-1)


class DataDirGenerator:

    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.npz_files = sorted(glob(data_dir + '/*.npz'))
        self.batch_size = batch_size
        n_files = len(self.npz_files)

        # Figure out how many total mini-batches are contained in the directory
        self.n_batches = 0
        for fi, f in enumerate(self.npz_files):
            print("Scanning {} ({}/{})".format(f, fi+1, n_files))
            self.n_batches += len(np.load(f)['board_tensors']) / batch_size
        print("Total mini-batches in {}: {}".format(data_dir, self.n_batches))

    @property
    def samples_per_epoch(self):
        return self.batch_size * self.n_batches

    def generate_samples(self):
        while True:
            for f in self.npz_files:
                data = np.load(f)
                board_tensors = data['board_tensors']
                extra_tensors = data['extra_tensors']
                target_tensors = data['target_tensors']
                for batch_idx in xrange(len(board_tensors) / self.batch_size):
                    start = batch_idx * self.batch_size
                    end = (batch_idx+1) * self.batch_size
                    yield (
                        [board_tensors[start:end], extra_tensors[start:end]],
                        target_tensors[start:end]
                    )

class ChessNet:

    def __init__(self, batch_size=4096):
        self.batch_size = batch_size
        self.board_score = self.initialize_model()

        repo = git.Repo(".")
        # if repo.is_dirty():
        #     print("Refusing to run with uncommitted changes. Please commit them first.")
        #     return
        savedir = 'logs/' + str(datetime.now()) + " " + repo.git.describe("--always", "--dirty", "--long")
        tbcb = TensorBoard(log_dir=savedir, histogram_freq=0, write_graph=True, write_images=False)
        mccb = ModelCheckpoint(savedir+'/model.{epoch:04d}-{val_loss:.2f}.hdf5',
                               monitor='val_loss', save_best_only=False)
        self.callbacks = [tbcb, mccb]


    def initialize_model(self):
        n = 1024
        n_pieces = len(Preprocessor.PIECES)

        # One hot encoding of the board, one class per piece type
        board_one_hot = Sequential()
        board_one_hot.add(Embedding(n_pieces, n_pieces, input_length=64, weights=[np.eye(n_pieces)], trainable=False))
        board_one_hot.add(Flatten())
        board_one_hot.add(BatchNormalization())

        # Encoding for extra board state (player turn, castling info, etc)
        extra = Sequential()
        extra.add(BatchNormalization(input_shape=(5, )))

        # Merge all inputs
        board_score = Sequential()
        merged_layer = Merge([board_one_hot, extra], mode='concat')
        board_score.add(merged_layer)

        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(BatchNormalization())
        board_score.add(Dropout(0.2))

        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(BatchNormalization())
        board_score.add(Dropout(0.2))

        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(BatchNormalization())
        board_score.add(Dropout(0.2))

        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(BatchNormalization())
        board_score.add(Dropout(0.2))

        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(BatchNormalization())
        board_score.add(Dropout(0.2))

        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
        board_score.add(BatchNormalization())
        board_score.add(Dropout(0.2))

        board_score.add(Dense(1, activation='sigmoid'))

        board_score.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', rel_acc])

        print(board_score.summary())

        return board_score

    def train_on_single_npz(self, npz_file):

        loaded = np.load(npz_file)
        board_tensors = loaded['board_tensors']
        extra_tensors = loaded['extra_tensors']
        target_tensors = loaded['target_tensors']
        print("Loaded from {}".format(npz_file))

        self.board_score.fit([board_tensors, extra_tensors], target_tensors,
                             nb_epoch=1000, callbacks=self.callbacks,
                             batch_size=self.batch_size, validation_split=0.1)

    def train_on_data_directory(self, data_dir):
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'validate')

        train_gen = DataDirGenerator(train_dir, self.batch_size)
        val_gen = DataDirGenerator(val_dir, self.batch_size)

        self.board_score.fit_generator(
            train_gen.generate_samples(), train_gen.samples_per_epoch,
            nb_epoch=1000, callbacks=self.callbacks,
            validation_data=val_gen.generate_samples(), nb_val_samples=val_gen.samples_per_epoch,
            max_q_size=1000, nb_worker=1, pickle_safe=True
        )


def main():
    net = ChessNet()
    net.train_on_data_directory(sys.argv[1])

if __name__ == '__main__':
    main()
