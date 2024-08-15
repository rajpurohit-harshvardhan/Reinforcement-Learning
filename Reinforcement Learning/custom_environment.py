import gym
import numpy as np
import random
from itertools import permutations


class CardPickingEnv(gym.Env):
    def __init__(self, cards_in_deck, cards_select_limit):
        super(CardPickingEnv, self).__init__()
        self.states_tuple_dict = None
        self.possible_states_dict = None
        self.cards_select_limit = cards_select_limit
        self.selected_cards_counter = 0
        self.states_list = self.generate_observation_space(cards_in_deck, cards_select_limit)
        self.state_to_index = {state: idx for idx, state in enumerate(self.states_list)}
        self.number_of_states = len(self.states_list)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.number_of_states,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.generate_possible_next_states_dict()  # generates a dictionary with list of possible states for each state
        self.current_state = None

    def step(self, action):
        current_index = np.argmax(self.current_state)
        current_state = self.states_list[current_index]
        possible_next_states = self.possible_states_dict[current_state]
        done = False
        reward = 0
        if len(possible_next_states[action]) == 0:
            return current_index, 0, True, {}

        next_state = random.choice(possible_next_states[action])

        selected_cards_current_state = [int(x) for x in
                                        current_state[current_state.index("|") + 1:current_state.index(")")] if
                                        x.isnumeric()]
        selected_cards_next_state = [int(x) for x in next_state[next_state.index("|") + 1:next_state.index(")")] if
                                     x.isnumeric()]

        reward = sum(selected_cards_next_state) - sum(selected_cards_current_state)

        if action == 1: # (pick)
            self.selected_cards_counter += 1

        if self.selected_cards_counter == self.cards_select_limit:
            done = True

        # reward = reward * 0.1
        return self.state_to_index[next_state], reward, done, {}

    def reset(self):
        self.current_state = np.zeros(len(self.states_list))
        random_state = self.states_list[random.randrange(self.number_of_states)]
        self.current_state[self.state_to_index[random_state]] = 1
        current_index = np.argmax(self.current_state)
        selected_state = self.states_list[current_index]
        selected_cards = [int(x) for x in
                          selected_state[selected_state.index("|") + 1:selected_state.index(")")] if
                          x.isnumeric()]
        self.selected_cards_counter = len(selected_cards)
        return self.state_to_index[random_state]

    def render(self, mode='human'):
        result = 1
        # print('render')

    def generate_observation_space(self, cards_in_deck, cards_select_limit):
        states_str = ''
        self.states_tuple_dict = {}
        for i in range(cards_select_limit + 1):
            states_tuple = list(
                permutations(cards_in_deck, i + 1)
            )
            states_str += ';'.join(map(str, states_tuple))
            self.states_tuple_dict[i] = {"states": ';'.join(map(str, states_tuple))}
            states_str += ';'

        states = []
        for state in states_str.split(';'):
            # handles the case when ; is at the end of the string and the last element is just an empty string.
            if len(state) > 0:
                # print("Appended state:", i.replace(",", "|", 1))
                state = state.replace(",", "|", 1)
                states.append(state)
        return states

    def generate_possible_next_states_dict(self):
        self.possible_states_dict = {}
        for index in self.states_tuple_dict:
            states_curr = [s.replace(",", "|", 1)
                           for s in self.states_tuple_dict[index]["states"].split(';')]
            states_next = []
            if index + 1 < len(self.states_tuple_dict.keys()):
                states_next = [s.replace(",", "|", 1)
                               for s in self.states_tuple_dict[index + 1]["states"].split(';')]

            for i in range(0, len(states_curr)):
                current_state = {
                    "lhs": states_curr[i][states_curr[i].index("(") + 1:states_curr[i].index("|")],
                    "rhs": states_curr[i][states_curr[i].index("|") + 1:states_curr[i].index(")")]
                }
                self.possible_states_dict[states_curr[i]] = {0: [], 1: [], 2: []} # 0:discard, 1: swap, 2: pick

                # Loop associates the current states with states in the same set to identify DISCARD action on state
                for j in range(0, len(states_curr)):
                    possible_state = {
                        "lhs": states_curr[j][states_curr[j].index("(") + 1:states_curr[j].index("|")],
                        "rhs": states_curr[j][states_curr[j].index("|") + 1:states_curr[j].index(")")]
                    }

                    if (current_state["lhs"].strip() != possible_state["lhs"].strip()
                            and current_state["rhs"].strip() == possible_state["rhs"].strip()):
                        self.possible_states_dict[states_curr[i]][0].append(states_curr[j])

                    cards_swapped = self.swap_card_states(current_state, possible_state)
                    if cards_swapped:
                        self.possible_states_dict[states_curr[i]][2].append(states_curr[j])

                # Loop associates the current states with states in the same set to identify SWAP action on state
                for k in range(0, len(states_next)):
                    possible_state = {
                        "lhs": states_next[k][states_next[k].index("(") + 1:states_next[k].index("|")],
                        "rhs": states_next[k][states_next[k].index("|") + 1:states_next[k].index(")")]
                    }

                    string1 = current_state["rhs"].strip()
                    result_string = string1 + ", " + current_state["lhs"].strip() \
                        if len(string1) > 0 else current_state["lhs"].strip()

                    if possible_state["rhs"].strip() == result_string:
                        self.possible_states_dict[states_curr[i]][1].append(states_next[k]) # current = (1|2) # (x|2,1)

                    selected_cards = [int(x) for x in possible_state["rhs"] if x.isnumeric()]
                    card_in_hand = int(possible_state["lhs"])
                    card_to_be_swapped = sorted(selected_cards)[0] if len(selected_cards) > 0 else 0
                    selected_cards[selected_cards.index(card_to_be_swapped)] = card_in_hand

    def swap_card_states(self, current_state, possible_state):
        # Handles the case when there is no selected cards in the hand
        if len(current_state["rhs"]) == 0:
            # print("Empty SPace, Not comparing")
            return False

        selected_cards = [int(x) for x in current_state["rhs"] if x.isnumeric()]
        card_to_be_swapped = sorted(selected_cards)[0] if len(selected_cards) > 0 else 0
        card_in_hand = int(current_state["lhs"])
        result_string = current_state["rhs"]
        index = result_string.index(str(card_to_be_swapped))
        result_string = result_string[:index] + str(card_in_hand) + result_string[index + 1:]
        if (possible_state["rhs"].strip() == result_string.strip() and possible_state["lhs"].strip() == str(
                card_to_be_swapped).strip()):
            # print("state found", possible_state)
            return True
