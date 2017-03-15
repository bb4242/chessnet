import chess
import chess.pgn
import chess.svg

import numpy as np
import os
import multiprocessing
from time import time
import queue
import pickle
import sys

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

class Preprocessor:

    PIECES = [None] + [chess.Piece(piece_type, color)
                       for color in [True, False]
                       for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                          chess.ROOK, chess.QUEEN, chess.KING]]
    PIECE_DICT = {piece: idx for idx, piece in enumerate(PIECES)}

    def __init__(self, pgn_file):
        self.pgn_file = pgn_file

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
        for w in workers:
            w.start()

        # Read PGN file
        game_data = []
        pgn_file = open(self.pgn_file)
        games_in = 0
        next_print = time()
        start_time = next_print
        while True:
            now = time()
            if now > next_print:
                next_print = now + 1
                elapsed_time = now - start_time
                gps = processed_game_count.value / elapsed_time
                print("Processed {:8d} games in {:8.1f} sec ({:8.1f} games/sec)".format(processed_game_count.value, elapsed_time, gps))

            while True:
                try:
                    game_data.append(result_queue.get_nowait())
                except queue.Empty:
                    break

            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            if len(game.errors) == 0 and 'SetUp' not in game.headers and len(list(game.main_line())) > 0:
                game_queue.put(game)
                games_in += 1

        # Tell workers to shut down
        [game_queue.put(None) for _ in range(len(workers))]
        game_queue.join()
        print "SHUTDOWN"

        # Aggregate data and write to file
        for _ in xrange(games_in - len(game_data)):
            game_data.append(result_queue.get())

        assert len(game_data) == games_in
        print("Total games: {}".format(games_in))

        board_tensors = np.concatenate([gd[0] for gd in game_data])
        extra_tensors = np.concatenate([gd[1] for gd in game_data])
        target_tensors = np.concatenate([gd[2] for gd in game_data])

        n_samples = len(board_tensors)
        permutation = np.random.permutation(n_samples)
        board_tensors = board_tensors[permutation]
        extra_tensors = extra_tensors[permutation]
        target_tensors = target_tensors[permutation]

        save_file = os.path.splitext(self.pgn_file)[0] + '-moves.npz'
        np.savez_compressed(save_file, board_tensors=board_tensors,
                            extra_tensors=extra_tensors,
                            target_tensors=target_tensors)

        print("Wrote {}".format(save_file))
        print("Total moves: {}".format(n_samples))
        print("Good/bad move ratio (should be 0.5): {}".format(np.mean(target_tensors)))
