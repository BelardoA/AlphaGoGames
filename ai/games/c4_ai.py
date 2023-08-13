import torch
torch.manual_seed(0)
from ai.games.connectfour import ConnectFour
from ai.models.torches import ResNet
import numpy as np
from ai.alpha_zero.mcts import MCTS

RUN = "train"


if __name__ == "__main__":
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if RUN == "plot":
        state = game.get_initial_state()
        state = game.get_next_state(state, 2, -1)
        state = game.get_next_state(state, 4, -1)
        state = game.get_next_state(state, 6, 1)
        state = game.get_next_state(state, 8, 1)

        encoded_state = game.get_encoded_state(state)

        tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

        model = ResNet(game, 4, 64, device=device)
        model.load_state_dict(torch.load('model_2.pt', map_location=device))
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

    elif RUN == "train":
        model = ResNet(game, 9, 128, device)
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        player = 1

        args = {
            'C': 2,
            'num_searches': 20,
            'eps': 0.0,
            'alpha': 0.3,
        }

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
