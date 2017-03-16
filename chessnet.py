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

from preprocessor import Preprocessor


class ChessNet:

    def __init__(self):
        self.board_score = self.initialize_model()

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
                             batch_size=4096, validation_split=0.1)


def main():
    #pp = Preprocessor('gorgobase-2500.pgn')
    #pp.process_pgn_file()

    net = ChessNet()
    net.train_on_games('gorgobase-2500-moves.npz')

if __name__ == '__main__':
    main()
