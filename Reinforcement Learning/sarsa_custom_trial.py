## KINDLY IGNORE THIS FILE
# Just trying out different stuff
#

import gym
import math
import itertools

class CardPickingEnv(gym.Env):
    def __init__(self, config):
        super(CardPickingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)

        num_unique_cards = 4
        num_total_values = 13
        # self.num_states = num_unique_cards * num_total_values
        self.num_total_values = 21

        self.num_cards = 7
        self.max_cards_selected = 3
        self.num_states = sum(math.comb(self.num_cards, i) for i in range(self.max_cards_selected + 1)) * (self.num_total_values + 1)

        # Define observation space
        self.observation_space = gym.spaces.Discrete(self.num_states)

        self.max_selected_cards = 3
        self.reset()

    def reset(self):
        # reset reward value
        self.selected_cards = []
        self.total_value = 0
        self.current_card = 0
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        done = False
        reward = 0

        if action == 1:  # Keep the card
            self.selected_cards.append(self.current_card + 1)
            self.total_value += self.current_card + 1

        self.current_card += 1

        if len(self.selected_cards) == self.max_cards_selected or self.current_card == self.num_cards:
            done = True
            reward = self.total_value

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        selected_tuple = tuple(sorted(self.selected_cards))
        state_index = self._get_state_index(selected_tuple, self.total_value)
        return state_index

    def _get_state_index(self, selected_tuple, total_value):
        # Map the selected cards and total value to a unique index
        card_combinations = list(itertools.combinations(range(1, self.num_cards + 1), len(selected_tuple)))
        combination_index = card_combinations.index(selected_tuple)
        return combination_index * (self.num_total_values + 1) + total_value
    def render(self):
        print(f"Selected Cards: {self.selected_cards}")
        print(f"Total Value: {self.total_value}")

    def close(self):
        super().close()
