"""AlphaZero algorithm implementation"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from ai.models.self_play_game import SPG
from ai.alpha_zero.mcts_parallel import MCTSParallel


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfPlay(self):
        return_memory = []
        player = 1
        self_play_games = [SPG(self.game) for _ in range(self.args['num_parallel_games'])]

        while len(self_play_games) > 0:
            states = np.stack([spg.state for spg in self_play_games])

            neutral_states = self.game.change_perspective(states, player)
            self.mcts.search(neutral_states, self_play_games)

            for i in range(len(self_play_games))[::-1]:
                spg = self_play_games[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs = action_probs / np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neuetral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neuetral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del self_play_games[i]
            player = self.game.get_opponent(player)
        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        for batchIndex in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIndex:min(len(memory) - 1, batchIndex + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()

            self.model.train()
            for _ in trange(self.args['num_epochs']):
                self.train(memory)

            for directory in ["training_models", "optimizers"]:
                if not os.path.exists(f"./{directory}/{self.game}"):
                    os.makedirs(f"./{directory}/{self.game}")

            torch.save(self.model.state_dict(), f"./training_models/{self.game}/{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"./optimizers/{self.game}/{iteration}.pt")





