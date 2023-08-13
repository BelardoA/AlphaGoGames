# AlphaGoGames
Games of tic tac toe and connect four in terminal with AlphaGo trained models.

## Setup
1. Install Python 3.9 or higher
2. Install the required packages with `pip install -r requirements.txt`
3. Run `python main.py`

## Usage
#### Training: _You can train your own models by running main, selecting the game of your choice and entering training mode._
- Run `python main.py`
- Select the game you would like to train a model for
  - `t` for Tic Tac Toe
  - `c` for Connect Four
- Enter `t` for training mode
- If you would like to change the # of training update `args` parameters, look at `./ai/games/c4_agi.py` for Connect Four and `./ai/games/t3_agi.py` for Tic Tac Toe.

#### Playing: _You can play against the trained models by running main, selecting the game of your choice and entering playing mode._
- Run `python main.py`
- Select the game you would like to play
  - `t` for Tic Tac Toe
  - `c` for Connect Four
- Enter `p` for playing mode
- Select player 1 and player 2
  - `1` for first player
  - `2` for player 2


## Examples
```commandline
$ python main.py
$ Tick-Tac-Toe or Connect Four? (t/c): t
$ Play, Train, or Plot? (p/t/pl): p
$ Player 1 or Player 2? (1/2): 1
$ Playing against model #2. Good luck!   
$ [[0. 0. 0.]                            
$  [0. 0. 0.]                            
$  [0. 0. 0.]]                           
$ valid_moves [0, 1, 2, 3, 4, 5, 6, 7, 8]
$ 1:    
```