# chessnet
Chess AI player based on deep learning.  This is a work in progress.  The basic idea is to train a deep neural network to distinguish between good and bad moves, then use this network to direct the selection of moves during the game.  Good moves are drawn from a database of historical chess matches played by expert players.  Bad moves are found by randomly selecting other legal moves that the expert players elected not to play from the same position.

## Database

A large set game database is required to train the model. So far, I've been using [gorgobase](http://gorgonian.weebly.com/pgn.html), which contains 2.8 million games.  This database is in [SCID](https://en.wikipedia.org/wiki/Shane%27s_Chess_Information_Database) format and will need to be converted to [PGN](https://en.wikipedia.org/wiki/Portable_Game_Notation).  The [SCID program](http://scid.sourceforge.net/) can do this.

## Currently implemented functionality

1. Parse a database of games into a structured set of training data.
1. Train a deep network on the training data.

## TODO

1. Use the trained model to select moves at game time.
1. Implement [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) and other techniques from the [AlphaGo paper](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
