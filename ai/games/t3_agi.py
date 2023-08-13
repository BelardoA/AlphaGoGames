import torch
torch.manual_seed(0)
from ai.games.tictactoe import TicTacToe
from ai.models.torches import ResNet
from ai.alpha_zero.mcts import MCTS
from ai.alpha_zero.alpha_zero_parallel import AlphaZeroParallel
import numpy as np
from random import randint


def play_tictactoe(mode: str, player: int):
    game = TicTacToe()
    random_model = randint(0, 9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "plot":
        state = game.get_initial_state()
        state = game.get_next_state(state, 2, -1)
        state = game.get_next_state(state, 4, -1)
        state = game.get_next_state(state, 6, 1)
        state = game.get_next_state(state, 8, 1)

        encoded_state = game.get_encoded_state(state)

        tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

        model = ResNet(game, 4, 64, device=device)
        model.load_state_dict(torch.load(f'./models/{game}/{random_model}.pt', map_location=device))
        model.eval()

        policy, value = model(tensor_state)
        value = value.item()
        policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

        print(value)
        print(state)
        print(tensor_state)

        import matplotlib.pyplot as plt
        plt.bar(range(game.action_size), policy)
        plt.show()

    elif mode == "train":
        model = ResNet(game, 4, 64, device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        args = {
            'C': 2,
            'num_searches': 600,
            'num_iterations': 10,
            'num_selfPlay_iterations': 500,
            'num_parallel_games': 1000,
            'num_epochs': 4,
            'batch_size': 64,
            "temperature": 1.25,
            "eps": 0.25,
            "alpha": 0.3
        }

        alpha_zero = AlphaZeroParallel(model, optimizer, game, args)
        alpha_zero.learn()
    elif mode == "play":
        args = {
            'C': 2,
            'num_searches': 100,
            'eps': 0.0,
            'alpha': 0.3,
        }

        model = ResNet(game, 4, 64, device)
        model.load_state_dict(torch.load(f'./ai/training_models/{game}/{random_model}.pt', map_location=device))
        model.eval()
        print(f"Playing against model #{random_model}. Good luck!")

        mcts = MCTS(game, args, model)
        state = game.get_initial_state()

        while True:
            print(state)

            if player == 1:
                valid_moves = game.get_valid_moves(state)
                print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
                action = int(input(f"{player}: "))

                if valid_moves[action] == 0:
                    print("invalid move")
                    continue
            else:
                neutral_state = game.change_perspective(state, player)
                mcts_probs = mcts.search(neutral_state)
                action = np.argmax(mcts_probs)

            state = game.get_next_state(state, action, player)

            value, is_terminal = game.get_value_and_terminated(state, action)

            if is_terminal:
                print(state)
                if value == 1:
                    print(f"{player} won")
                else:
                    print("draw")
                break

            player = game.get_opponent(player)
