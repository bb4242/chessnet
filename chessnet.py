#!/usr/bin/python

from datetime import datetime

import chess
import chess.pgn
import chess.svg

import git
import os

import numpy as np

from keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout, Lambda, merge, Merge, Embedding
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K

from preprocessor import Preprocessor


class ChessNet:

    def __init__(self):
        self.board_score = self.initialize_model()

    def initialize_model(self):
        n = 1024

        # uint8 is ok for <= 256 classes, otherwise use int32
        #board_score.add(Input(shape=input_shape, dtype='uint8'))

        # Without the output_shape, Keras tries to infer it using calling the function
        # on an float32 input, which results in error in TensorFlow:
        #
        #   TypeError: DataType float32 for attr 'TI' not in list of allowed values: uint8, int32, int64

        board_one_hot = Sequential()
        nb_classes = len(Preprocessor.PIECES)
        input_shape = (64, )
        output_shape = (64, nb_classes)
        board_one_hot.add(Lambda(K.one_hot,
                                 arguments={'nb_classes': nb_classes},
                                 input_shape=input_shape, input_dtype='uint8',
                                 output_shape=output_shape))
        board_one_hot.add(Flatten())
        board_one_hot.add(BatchNormalization())

        extra = Sequential()
        extra.add(BatchNormalization(input_shape=(5, )))
#        extra.add(Dense(32, input_shape=(5, )))

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

        board_score.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(board_score.summary())

        return board_score

    def train_on_games(self, npz_file):

        loaded = np.load(npz_file)
        board_tensors = loaded['board_tensors']
        extra_tensors = loaded['extra_tensors']
        target_tensors = loaded['target_tensors']
        print("Loaded from {}".format(npz_file))

        repo = git.Repo(".")
        if repo.is_dirty():
            print("Refusing to run with uncommitted changes. Please commit them first.")
            return

        savedir = 'logs/' + str(datetime.now()) + " " + repo.git.describe("--always", "--dirty", "--long")
        tbcb = TensorBoard(log_dir=savedir, histogram_freq=0, write_graph=True, write_images=False)
        mccb = ModelCheckpoint(savedir+'/model.{epoch:04d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True)
        cb = [tbcb, mccb]

        self.board_score.fit([board_tensors, extra_tensors], target_tensors, nb_epoch=1000, callbacks=cb,
                             batch_size=4096*2, validation_split=0.1)


def main():
    #pp = Preprocessor('gorgobase-2500.pgn')
    #pp.process_pgn_file()

    net = ChessNet()
    net.train_on_games('gorgobase-2500-moves.npz')

if __name__ == '__main__':
    main()
