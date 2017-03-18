#!/usr/bin/python

import chess
import chess.pgn

import numpy as np
import os
import multiprocessing
from time import time
import queue
import sys
from glob import glob
import shutil

# This is necessary to prevent pickling errors when sending chess.Game objects to the workers
sys.setrecursionlimit(20000)


class GameProcessWorker(multiprocessing.Process):

    def __init__(self, game_queue, result_queue, processed_game_count):
        multiprocessing.Process.__init__(self)
        self.game_queue = game_queue
        self.result_queue = result_queue
        self.processed_game_count = processed_game_count

    def run(self):
        while True:
            next_game = self.game_queue.get()
            if next_game is None:
                self.game_queue.task_done()
                break
            result = Preprocessor.process_game(next_game)
            self.result_queue.put(result)
            with self.processed_game_count.get_lock():
                self.processed_game_count.value += 1
            self.game_queue.task_done()

class Accumulator:
    """Accumulates tensor results and writes them to disk in chunks of constant size"""

    def __init__(self, output_dir, chunk_size=2**20):
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        os.makedirs(output_dir)

        self.cur_idx = 0
        self.file_idx = 0
        self.board_tensors = np.zeros((chunk_size, 64), dtype='uint8')
        self.extra_tensors = np.zeros((chunk_size, 5),  dtype='uint8')
        self.target_tensors = np.zeros(chunk_size, dtype='uint8')

    def push(self, res_board, res_extra, res_target):
        n_avail = len(res_board)
        n_pushed = min(self.chunk_size - self.cur_idx, n_avail)

        i = self.cur_idx
        j = self.cur_idx + n_pushed
        self.board_tensors[i:j] = res_board[:n_pushed]
        self.extra_tensors[i:j] = res_extra[:n_pushed]
        self.target_tensors[i:j] = res_target[:n_pushed]
        self.cur_idx += n_pushed

        if self.cur_idx == self.chunk_size:
            self.flush()

        if n_avail - n_pushed > 0:
            self.push(res_board[n_pushed:], res_extra[n_pushed:], res_target[n_pushed:])

    def flush(self):
        self.file_idx += 1

        # Shuffle data
        permutation = np.random.permutation(self.cur_idx)
        board_tensors = self.board_tensors[permutation]
        extra_tensors = self.extra_tensors[permutation]
        target_tensors = self.target_tensors[permutation]

        # Write file
        save_file = os.path.join(self.output_dir, '{:06d}.npz'.format(self.file_idx))
        np.savez_compressed(save_file, board_tensors=board_tensors,
                            extra_tensors=extra_tensors,
                            target_tensors=target_tensors)

        print("Wrote {} moves to {}".format(len(board_tensors), save_file))
        self.cur_idx = 0


class Preprocessor:

    PIECES = [None] + [chess.Piece(piece_type, color)
                       for color in [True, False]
                       for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                          chess.ROOK, chess.QUEEN, chess.KING]]
    PIECE_DICT = {piece: idx for idx, piece in enumerate(PIECES)}

    def __init__(self, pgn_file, chunk_size=2**20, train_frac=0.8, val_frac=0.1):
        """
        :param pgn_file: Path to the PGN file to process
        :param chunk_size: Number of moves to write per file
        :param train_frac, val_frac: Fraction of the data to use for training and validation sets.
        If these sum to less than 1, an additional test set will be created
        from the remainder
        """
        self.pgn_file = pgn_file
        self.chunk_size = chunk_size
        self.train_frac = train_frac
        self.val_frac = val_frac

    @classmethod
    def board_to_tensor(cls, board):
        board_tensor = np.array([cls.PIECE_DICT[board.piece_at(square)] for square in chess.SQUARES], dtype='uint8')
        extra_state_tensor = np.array([
            board.turn,
            board.has_kingside_castling_rights(True), board.has_queenside_castling_rights(True),
            board.has_kingside_castling_rights(False), board.has_queenside_castling_rights(False)
        ], dtype='uint8')
        return board_tensor, extra_state_tensor

    @classmethod
    def process_game(cls, game):
        b = chess.Board()
        board_tensors = []
        extra_tensors = []
        targets = []

        def _add(move, target):
            b.push(move)
            bt, et = cls.board_to_tensor(b)
            board_tensors.append(bt)
            extra_tensors.append(et)
            targets.append(target)
            b.pop()

        for selected_move in game.main_line():
            # Get a random legal move that was not selected
            legal_moves = np.array(list(b.legal_moves))
            np.random.shuffle(legal_moves)
            for legal_move in legal_moves:
                if legal_move != selected_move:
                    _add(legal_move, 0)
                    _add(selected_move, 1)
                    break
            b.push(selected_move)
        return np.array(board_tensors), np.array(extra_tensors), np.array(targets)

    def process_pgn_file(self):

        # Start workers
        game_queue = multiprocessing.JoinableQueue(500)
        processed_game_count = multiprocessing.Value('i', 0)
        result_queue = multiprocessing.Queue(500)
        workers = [GameProcessWorker(game_queue, result_queue, processed_game_count)
                   for i in range(multiprocessing.cpu_count())]
        [w.start() for w in workers]

        # Read PGN file games in a random order
        pgn_file = open(self.pgn_file)
        print("Reading PGN game offsets")
        game_offsets = np.array(list(chess.pgn.scan_offsets(pgn_file)))
        np.random.shuffle(game_offsets)
        n_games = len(game_offsets)
        print("Found {} games".format(n_games))

        moves_dir = os.path.splitext(self.pgn_file)[0] + '-moves'
        move_acc = Accumulator(moves_dir, self.chunk_size)
        games_in = 0
        games_in_flight = 0
        next_print = time()
        start_time = next_print

        for offset in game_offsets:
            now = time()
            if now > next_print:
                next_print = now + 1
                elapsed_time = now - start_time
                gps = processed_game_count.value / elapsed_time
                print("Processed {:8d} games in {:8.1f} sec ({:8.1f} games/sec)".format(processed_game_count.value, elapsed_time, gps))

            # Accumulate and write to disk already processed results
            while True:
                try:
                    move_acc.push(*result_queue.get_nowait())
                    games_in_flight -= 1
                except queue.Empty:
                    break

            # Read and queue the next game for processing
            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)
            if len(game.errors) == 0 and 'SetUp' not in game.headers and len(list(game.main_line())) > 0:
                game_queue.put(game)
                games_in += 1
                games_in_flight += 1

        # Accumulate remaining games in flight
        for _ in xrange(games_in_flight):
            move_acc.push(*result_queue.get())

        # Flush any remaining moves to disk
        move_acc.flush()

        # Move files into train/validate/test sets
        npz_files = sorted(glob(moves_dir + '/*.npz'))
        n_files = len(npz_files)
        train_idx = int(n_files*self.train_frac)
        val_idx = train_idx + int(n_files*self.val_frac)

        dirs = [os.path.join(moves_dir, d) for d in ['train', 'validate', 'test']]
        [os.makedirs(d) for d in dirs]
        set_files = np.split(npz_files, [train_idx, val_idx])
        for dest_dir, files in zip(dirs, set_files):
            for f in files:
                shutil.move(f, dest_dir)

        # Tell workers to shut down
        [game_queue.put(None) for _ in range(len(workers))]
        game_queue.join()
        [w.join() for w in workers]


def main():
    Preprocessor(sys.argv[1]).process_pgn_file()

if __name__ == '__main__':
    main()
