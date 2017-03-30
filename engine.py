from abc import ABCMeta, abstractmethod
import multiprocessing

import chess
import chess.uci


class ChessEngine:
    """Abstract class that encapsulates a chess playing engine"""
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def new_game(self):
        """Resets the engine to a new game state"""
        pass

    @abstractmethod
    def select_move(self, board):
        """Generate a move given the current board position.
        Should return (selected_move, extra_info) where
        selected_move is a chess.Move and extra_info is a dict
        of any extra info the engine wishes to return (position
        evaluation scores, etc)
        """


class StockfishEngine(ChessEngine):

    def __init__(self, skill=20, cpu_count=multiprocessing.cpu_count(),
                 search_opts={'nodes': 10000000}):
        """Initialize the stockfish engine.
        skill is the skill level of the engine, between 0 (worst) and 20 (best)
        cpu_count: The number of CPUs to use when searching for the next move
        search_opts: Options to pass to chess.uci.Engine.go(). depth, nodes, movetime
        params can be used to control how long to search for
        """
        self.engine = chess.uci.popen_engine('stockfish')
        self.engine.uci()
        self.engine.setoption({"Threads": cpu_count})
        self.info_handler = chess.uci.InfoHandler()
        self.engine.info_handlers.append(self.info_handler)
        self.search_opts = search_opts
        self.set_skill(skill)
        self.new_game()

    def set_skill(self, skill=20):
        """Set the skill level of the engine
        Range: 0 (worst) to 20 (best)
        """
        self.engine.setoption({"Skill Level": skill})

    def new_game(self):
        self.engine.ucinewgame()

    def select_move(self, board):
        self.engine.position(board)
        move = self.engine.go(**self.search_opts).bestmove
        score = self.info_handler.info['score']
        return move, score


class EngineMatchup:
    """Plays two engines against each other"""

    def __init__(self, engine1, engine2):
        self.engines = [engine1, engine2]

    def play_match(self, engine1_white=True):
        """Play a match between the two engines, and return the result
        from engine1's perspective: 1 for a win, 0 for a draw, -1 for a loss
        """
        [engine.new_game() for engine in self.engines]
        board = chess.Board()
        engine_to_move_idx = 0 if engine1_white else 1

        while not board.is_game_over():
            print self.engines[engine_to_move_idx]
            move, extra = self.engines[engine_to_move_idx].select_move(board)
            san = board.san(move)
            board.push(move)
            engine_to_move_idx = (engine_to_move_idx + 1) % 2

            print board.fullmove_number, san
            print extra
            print board
            print

        result_str = board.result()
        if result_str == '1-0':
            result = 1
        elif result_str == '0-1':
            result = -1
        else:
            result = 0

        if not engine1_white:
            result *= -1

        return result
