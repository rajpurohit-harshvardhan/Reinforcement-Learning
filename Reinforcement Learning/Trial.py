## KINDLY IGNORE THIS FILE
# Just trying out different stuff
#

from itertools import permutations
import numpy as np
import random

action_space = {
    'n': 3,
    'possible_states': [0, 1, 2],  # where 0 indicates DISCARD ; 1 indicates SWAP ; 2 indicates PICK
}
observation_space = {
    'n': 0,
    'possible_states': [],
}

# inputs
cards_in_deck = [1, 2, 3]
cards_select_limit = 2

# Using variables for now ; can be replaced by object of observation space
states = []
number_of_states = 0

states_str = ''
# r = cards limit to be selected
# r+1 is used here because the representation of state is a combination of :
# cards to be selected (r) + an additional var to denote current card in hand that takes in 1 var therefore r+1
states_tuple_dict = {}
for i in range(cards_select_limit + 1):
    # I+1 is used because I symbolizes number of cards selected,
    # Initially, Even though we are not selecting a card, we are switching values for variable denoting card in hand.
    # for i>0, 1 additional slot is required to have possible combination for variable denoting card in hand.

    states_tuple = list(
        permutations(cards_in_deck, i + 1)
    )  # returns a tuple type, thus needs to be converted to string
    states_str += ';'.join(map(str, states_tuple))
    states_tuple_dict[i] = {"states": ';'.join(map(str, states_tuple)), "start": number_of_states, "end": len(states_tuple)}
    states_str += ';'  # to separate the states ; if removed, 2 states in the middle gets clubbed together
    number_of_states += len(states_tuple)
    # for i in combinations:
    # state_str = ','.join(map(str, i))
    # state_str.replace(",", '|', 1)
    # print(state_str)
    # print(i)

print(number_of_states, states_str)  # contains the string of states seperated by ;

for state in states_str.split(';'):
    if len(state) > 0:  # handles the case when ; is at the end of the string and the last element is just an empty string.
        # print("Appended state:", i.replace(",", "|", 1))
        state = state.replace(",", "|", 1)
        selected_cards = [int(x) for x in state[state.index("|") + 1:state.index(")")] if x.isnumeric()]
        sum_of_selected_cards = sum(selected_cards)
        card_in_hand = int(state[state.index("(") + 1:state.index("|")][0])
        states.append(state)
print(len(states), states)


# There are 2 methods to create the Q table
# first :: 1 table dimensions : States X Actions :: cells indicate the
# reward gain when chosen a particular action on a given state
# second :: 2 tables with dimensions : States X States
# 1 table indicates what are the possible states that the agent would be able to propagate to for a given state
# 2 table indicates the reward associated with the propagation

def prepare_q_table_first_method(states_str):
    Q = np.zeros((number_of_states, action_space['n']))

    states_str_list = states_str.split(';')
    for i in range(0, len(states_str_list)):
        state = states_str_list[i].replace(",", "|", 1)
        if len(state) == 0:
            break
        selected_cards = [int(x) for x in state[state.index("|") + 1:state.index(")")] if x.isnumeric()]
        card_in_hand = int(state[state.index("(") + 1:state.index("|")])

        card_to_be_swapped = sorted(selected_cards)[0] if len(selected_cards) > 0 else 0

        # first method
        # for DISCARD action ::
        # idea was that we are essentially getting rid of a card and further possibilities
        # Q[i][0] = sum(selected_cards) - card_in_hand
        sum_of_selected_cards = sum(selected_cards)
        # Q[i][0] = sum_of_selected_cards - card_to_be_swapped + card_in_hand
        if card_to_be_swapped < card_in_hand:
            # Q[i][0] = sum_of_selected_cards - card_in_hand
            # SWAP card into selected part of the hands
            Q[i][0] = sum_of_selected_cards - card_to_be_swapped + card_in_hand
        else:
            Q[i][0] = sum_of_selected_cards

        if len(selected_cards) < cards_select_limit and card_to_be_swapped != 0:
            selected_cards_list = selected_cards[:]
            selected_cards_list.append(card_in_hand)
            sum_of_selected_cards = sum(selected_cards_list)
            Q[i][2] = sum_of_selected_cards

        # SWAP the card_in_hand with the lowest one in selected_cards
        if card_to_be_swapped == 0:
            selected_cards.append(card_in_hand)
        else:
            selected_cards[selected_cards.index(card_to_be_swapped)] = card_in_hand
        sum_of_selected_cards = sum(selected_cards)
        Q[i][1] = sum_of_selected_cards

        # print(Q[i])
    return Q


# q_table_first_method = prepare_q_table_first_method(states_str)
states_list = []
for state in states_str.split(';'):
    # handles the case when ; is at the end of the string and the last element is just an empty string.
    if len(state) > 0:
        # print("Appended state:", i.replace(",", "|", 1))
        state = state.replace(",", "|", 1)
        states_list.append(state)

state_to_index = {state: idx for idx, state in enumerate(states_list)}
# print(q_table_first_method)


def generate_possible_next_states_matrix(states_tuple_dict):
    I = np.zeros((number_of_states, number_of_states))
    possible_states = {}
    for index in states_tuple_dict:
        states_curr = [s.replace(",", "|", 1) for s in states_tuple_dict[index]["states"].split(';')]
        states_next = []
        if index+1 < len(states_tuple_dict.keys()):
            states_next = [s.replace(",", "|", 1) for s in states_tuple_dict[index+1]["states"].split(';')]

        for i in range(0, len(states_curr)):
            current_state = {
                    "lhs": states_curr[i][states_curr[i].index("(") + 1:states_curr[i].index("|")],
                    "rhs": states_curr[i][states_curr[i].index("|") + 1:states_curr[i].index(")")]
                }
            possible_states[states_curr[i]] = {
                0: [],
                1: []
            }
            for j in range(0, len(states_curr)):
                possible_state = {
                    "lhs": states_curr[j][states_curr[j].index("(") + 1:states_curr[j].index("|")],
                    "rhs": states_curr[j][states_curr[j].index("|") + 1:states_curr[j].index(")")]
                }

                if (current_state["lhs"].strip() != possible_state["lhs"].strip()
                        and current_state["rhs"].strip() == possible_state["rhs"].strip()):
                    possible_states[states_curr[i]][0].append(states_curr[j])

            for k in range(0, len(states_next)):
                possible_state = {
                    "lhs": states_next[k][states_next[k].index("(") + 1:states_next[k].index("|")],
                    "rhs": states_next[k][states_next[k].index("|") + 1:states_next[k].index(")")]
                }

                string1 = current_state["rhs"].strip()
                result_string = string1 + ", " + current_state["lhs"].strip() \
                    if len(string1) > 0 else current_state["lhs"].strip()

                if (possible_state["rhs"].strip() == result_string):
                    possible_states[states_curr[i]][1].append(states_next[k])

        # print(states_curr, states_next)
    return possible_states
    print(possible_states, states)


possible_states_dict = generate_possible_next_states_matrix(states_tuple_dict)

def take_step(action, current_state, selected_cards_counter):
    # current_index = np.argmax(current_state)
    current_index = state_to_index[current_state]
    current_state = states_list[current_index]
    possible_next_states = possible_states_dict[current_state]

    done = False
    reward = 0
    if len(possible_next_states[action]) == 0:
        return current_index, reward, True, {}

    next_state = random.choice(possible_next_states[action])

    selected_cards_current_state = [int(x) for x in
                                    current_state[current_state.index("|") + 1:current_state.index(")")] if
                                    x.isnumeric()]
    selected_cards_next_state = [int(x) for x in next_state[next_state.index("|") + 1:next_state.index(")")] if
                                 x.isnumeric()]

    if sum(selected_cards_current_state) < sum(selected_cards_next_state):
        reward = 1
    else:
        reward = -1

    if action == 1:
        selected_cards_counter += 1

    if selected_cards_counter == cards_select_limit:
        done = True

    return state_to_index[next_state], reward, done, {}

result = take_step(1, "(2| 3)", 0)
print(result)

def prepare_q_table_second_method(states_str):
    I = np.zeros((number_of_states, number_of_states))
    Q = np.zeros((number_of_states, number_of_states))
    for i in range(number_of_states):
        for j in range(number_of_states):
            print(1)
