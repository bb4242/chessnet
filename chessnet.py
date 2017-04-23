#!/usr/bin/python

from datetime import datetime
import sys
import os
from glob import glob
import click
import numpy as np
import git
import copy

from keras.layers import Dense, Convolution2D, Flatten, BatchNormalization, Dropout, Merge, Embedding, RepeatVector, Reshape, MaxPooling2D
from keras.models import load_model, Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K

from preprocessor import Preprocessor
from engine import ChessEngine


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

    def __init__(self, data_dirs, batch_size):
        self.npz_files = []
        for data_dir in data_dirs:
            self.npz_files += glob(data_dir + '/*.npz')
        self.npz_files = np.array(self.npz_files)
        np.random.shuffle(self.npz_files)

        self.batch_size = batch_size
        n_files = len(self.npz_files)

        # Figure out how many total mini-batches are contained in the directory
        self.n_batches = 0
        for fi, f in enumerate(self.npz_files):
            print("Scanning {} ({}/{})".format(f, fi+1, n_files))
            self.n_batches += len(np.load(f)['board_tensors']) / batch_size
        print("Total mini-batches in {}: {}".format(data_dirs, self.n_batches))

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

class ChessNet(ChessEngine):

    def __init__(self, batch_size=4096, load_model_filename=None, move_temp=0.05, log=True):
        self.batch_size = batch_size
        self.move_temp = move_temp

        if load_model_filename is not None:
            print("Loading saved model from {}".format(load_model_filename))
            self.name = os.path.basename(load_model_filename)
            self.board_score = load_model(load_model_filename, custom_objects={'rel_acc': rel_acc})
        else:
            self.name = ""
            self.board_score = self.initialize_model()

        try:
            repo = git.Repo(".")
            ver_str = repo.git.describe("--always", "--dirty", "--long")
        except:
            ver_str = 'unknown'

        if log:
            savedir = 'logs/' + str(datetime.now()) + " " + ver_str
            tbcb = TensorBoard(log_dir=savedir, histogram_freq=0, write_graph=True, write_images=False)
            mccb = ModelCheckpoint(savedir+'/model.{epoch:04d}-{loss:.4f}-{acc:.4f}-{rel_acc:.4f}-{val_loss:.4f}-{val_acc:.4f}-{val_rel_acc:.4f}.hdf5',
                                   monitor='val_loss', save_best_only=False)
            self.callbacks = [tbcb, mccb]
        else:
            self.callbacks = []


    def initialize_model(self):
        n = 64
        n_pieces = len(Preprocessor.PIECES)

        # One hot encoding of the board, one class per piece type
        board_one_hot = Sequential()
        board_one_hot.add(Embedding(n_pieces, n_pieces, input_length=64, weights=[np.eye(n_pieces)], trainable=False))
        board_one_hot.add(Reshape((8, 8, n_pieces)))

        # Encoding for extra board state (player turn, castling info, etc)
        extra = Sequential()
        extra.add(RepeatVector(64, input_shape=(5, )))
        extra.add(Reshape((8, 8, 5)))

        # Merge all inputs
        board_score = Sequential()
        board_score.add(Merge([board_one_hot, extra], mode='concat'))

        board_score.add(Convolution2D(n, 3, 3, activation='relu', border_mode='same'))
        board_score.add(Convolution2D(n, 3, 3, activation='relu', border_mode='same'))
        board_score.add(BatchNormalization())
#        board_score.add(Dropout(0.2))

        board_score.add(Convolution2D(n, 3, 3, activation='relu', border_mode='same'))
        board_score.add(Convolution2D(n, 3, 3, activation='relu', border_mode='same'))
        board_score.add(BatchNormalization())
#        board_score.add(Dropout(0.2))

#        board_score.add(MaxPooling2D((2, 2)))

        board_score.add(Convolution2D(n, 3, 3, activation='relu', border_mode='same'))
        board_score.add(Convolution2D(n, 3, 3, activation='relu', border_mode='same'))
        board_score.add(BatchNormalization())
#        board_score.add(Dropout(0.2))

        board_score.add(Convolution2D(n, 3, 3, activation='relu', border_mode='same'))
        board_score.add(Convolution2D(n, 3, 3, activation='relu', border_mode='same'))
        board_score.add(BatchNormalization())
#        board_score.add(Dropout(0.2))

        board_score.add(Convolution2D(n, 3, 3, activation='relu', border_mode='same'))
        board_score.add(Convolution2D(n, 3, 3, activation='relu', border_mode='same'))
        board_score.add(BatchNormalization())
#        board_score.add(Dropout(0.2))


        board_score.add(Flatten())
        board_score.add(Dense(n, activation='relu'))
        board_score.add(Dense(n, activation='relu'))
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
                             nb_epoch=1000, callbacks=self.callbacks, shuffle=False,
                             batch_size=self.batch_size, validation_split=0.1)

    def train_on_data_directories(self, data_dirs):
        train_dirs = [os.path.join(data_dir, 'train') for data_dir in data_dirs]
        val_dirs = [os.path.join(data_dir, 'validate') for data_dir in data_dirs]

        train_gen = DataDirGenerator(train_dirs, self.batch_size)
        val_gen = DataDirGenerator(val_dirs, self.batch_size)

        self.board_score.fit_generator(
            train_gen.generate_samples(), train_gen.samples_per_epoch,
            nb_epoch=1000, callbacks=self.callbacks,
            validation_data=val_gen.generate_samples(), nb_val_samples=val_gen.samples_per_epoch,
            max_q_size=1000, nb_worker=1, pickle_safe=True
        )

    def analyze_position(self, board):
        """Given a game board, analyze the position by evaluating the network on the legal moves
        Returns a list of (score, move) tuples, sorted from highest to lowest score, where
        the sum of scores is 1.
        """

        # Get the encoded boards reachable via legal moves from this position
        next_board_tensors = []
        next_extra_tensors = []
        moves = list(board.legal_moves)
        for move in moves:
            board.push(move)
            nbt, net = Preprocessor.board_to_tensor(board)
            board.pop()
            next_board_tensors.append(nbt)
            next_extra_tensors.append(net)
        next_board_tensors = np.array(next_board_tensors)
        next_extra_tensors = np.array(next_extra_tensors)

        # Evaluate the network on the reachable moves
        scores = self.board_score.predict([next_board_tensors, next_extra_tensors])[:, 0]

        # Convert scores to a probability distribution
        scores_dist = np.exp(scores/self.move_temp)
        scores_dist /= np.sum(scores_dist)

        return sorted(zip(scores_dist, moves), reverse=True)

    def get_board_score(self, board):
        """Evaluate the network on the current board and return the score"""
        nbt, net = Preprocessor.board_to_tensor(board)
        return self.board_score.predict([nbt[np.newaxis], net[np.newaxis]])[0, 0]

    def select_move(self, board, depth):
        """Choose a move for the current position, searching to the specified
        (2-ply) depth before applying the board score to evaluate leaf nodes.
        """
        self.n_nodes = 0
        self.alpha_cutoffs = 0
        self.beta_cutoffs = 0

        minimax_val, best_board = self.alphabeta(board, depth*2)

        info = {
            'minimax_val': minimax_val,
            'nodes_searched': self.n_nodes,
            'alpha_cutoffs': self.alpha_cutoffs,
            'beta_cutoffs': self.beta_cutoffs
        }
        return best_board.move_stack[0], info

    def alphabeta(self, board, depth, alpha=-np.inf, beta=np.inf):
        if depth == 0:
            self.n_nodes += 1
            if board.is_game_over():
                if board.result == '1-0':
                    val = np.inf
                elif board.result == '0-1':
                    val = -np.inf
                else:
                    val = 0
            else:
                val = self.get_board_score(board)
            return val, copy.deepcopy(board)

        moves = [e[1] for e in self.analyze_position(board)]

        if board.turn == chess.WHITE:
            v = -np.inf
            bb = None
            for move in moves:
                board.push(move)
                v0, bcopy = self.alphabeta(board, depth-1, alpha, beta)
                if v0 > v:
                    v = v0
                    bb = bcopy
                board.pop()
                alpha = max(v, alpha)
                if beta <= alpha:
                    self.beta_cutoffs += 1
                    break
            return v, bb
        else:
            v = np.inf
            bb = None
            for move in moves:
                board.push(move)
                v0, bcopy = self.alphabeta(board, depth-1, alpha, beta)
                if v0 < v:
                    v = v0
                    bb = bcopy
                board.pop()
                beta = min(v, beta)
                if beta <= alpha:
                    self.alpha_cutoffs += 1
                    break
            return v, bb


@click.command()
@click.option('--load', '-l', default=None, help='Start training with the weights saved in the given model')
@click.option('--log/--no-log', default=True, help="Don't write log file (for testing purposes)")
@click.argument('training_dirs', nargs=-1)
def main(load, log, training_dirs):
    net = ChessNet(2048, load, log=log)
    net.train_on_data_directories(training_dirs)

if __name__ == '__main__':
    main()
