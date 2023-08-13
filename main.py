from ai.games.c4_agi import play_connect_four
from ai.games.t3_agi import play_tictactoe

MODE: dict = {
    "p": "play",
    "t": "train",
    "pl": "plot",
}


if __name__ == '__main__':
    game_choice = input("Tick-Tac-Toe or Connect Four? (t/c): ")
    mode_choice = input("Play, Train, or Plot? (p/t/pl): ")
    if game_choice not in ["t", "c"]:
        raise ValueError("Invalid game choice.")
    if mode_choice not in ["p", "t", "pl"]:
        raise ValueError("Invalid mode choice.")
    while True:
        def play_again():
            return input("Play again? (y/n): ")
        if mode_choice == "p":
            player_choice = input("Player 1 or Player 2? (1/2): ")
            if player_choice not in ["1", "2"]:
                raise ValueError("Invalid player choice.")
        if game_choice.lower() == "t":
            play_tictactoe(MODE[mode_choice.lower()], int(player_choice if player_choice == "1" else -1))
        if game_choice.lower() == "c":
            play_connect_four(MODE[mode_choice.lower()], int(player_choice if player_choice == "1" else -1))
        retry = play_again()
        if retry not in ["y", "n"]:
            print("Invalid input.")
            play_again()
        if retry.lower() == "n":
            break
