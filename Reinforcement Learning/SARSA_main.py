import gym
import numpy as np
from custom_environment import CardPickingEnv
import tabulate
import random

epsilon = 0.5
total_episodes = 100000
max_steps = 100
alpha = 0.85
gamma = 0.95
# inputs
cards_in_deck = [1, 2, 3, 4, 5, 6, 7]
cards_select_limit = 3

env = CardPickingEnv(cards_in_deck, cards_select_limit)
Q = np.zeros((env.observation_space.shape[0], env.action_space.n))


def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def update(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    Q[state, action] = Q[state, action] + alpha * (target - predict)


for episode in range(total_episodes):
    t = 0
    state1 = env.reset()
    action1 = choose_action(state1)
    reward = 0
    print("Episode ::", episode)
    while t < max_steps:
        # Visualizing the training
        env.render()

        # Getting the next state
        state2, reward, done, info = env.step(action1)

        # Choosing the next action
        action2 = choose_action(state2)

        # Learning the Q-value
        update(state1, state2, reward, action1, action2)

        state1 = state2
        action1 = action2

        # Updating the respective values
        t += 1
        # reward += 1

        # If at the end of learning process
        if done:
            break

print("Performance : ", reward / total_episodes)

# Visualizing the Q-matrix
print(Q)

states = env.states_list
table = []
for i in range(len(states)):
    table.append([states[i], Q[i][0], Q[i][1], Q[i][2]])

print(tabulate.tabulate(table, headers=["States", "DISCARD", "PICK", "SWAP"]))


def calculate_reward(current_state, next_state):
    selected_cards_current_state = [int(x) for x in
                                    current_state[current_state.index("|") + 1:current_state.index(")")] if
                                    x.isnumeric()]
    selected_cards_next_state = [int(x) for x in next_state[next_state.index("|") + 1:next_state.index(")")] if
                                 x.isnumeric()]
    return sum(selected_cards_next_state) - sum(selected_cards_current_state)


print("PLAYING :::: ")
# INPUT :: shuffled deck of cards, select limit.
max_reward = sum(sorted(cards_in_deck)[-cards_select_limit:])
random.shuffle(cards_in_deck)
actions = {0: "DISCARD", 1: "PICK", 2: "SWAP"}

# WORKING ::
selected_cards = []
#  - pop out a card from the deck, represent it as a state, EVERY popped out card is the card in hand
card_in_hand = cards_in_deck.pop()
current_state = "(" + str(card_in_hand) + "|" + str(",".join(selected_cards)) + ")"
current_state_index = env.state_to_index[current_state]
reward = 0

loop = True
cards_selected_counter = 0
while loop:
    Q_list = Q[current_state_index].tolist()
    Q_list_reversed = sorted(Q_list)[::-1]

    for q_value in Q_list_reversed:
        action = Q_list.index(q_value)

        possible_states = env.possible_states_dict[current_state][action]
        if len(possible_states) == 0:
            continue

        state_selected = None
        reward_attained = 0
        for state in possible_states:
            calculated_reward = calculate_reward(current_state, state)
            if calculated_reward < 0:  # handles case when reward for given state and action pair is penalising
                continue

            reward_attained = calculated_reward
            state_selected = state
            env.possible_states_dict[current_state][action].remove(state_selected)
            break

        if not state_selected:  # handles the case when no state ,for given action, is rewarding
            continue

        reward += reward_attained
        print(current_state + " + "+ actions[action] + " = " + state_selected)
        current_state = state_selected
        current_state_index = env.state_to_index[state_selected]
        break

    selected_cards = [int(x) for x in
                      current_state[current_state.index("|") + 1:current_state.index(")")] if
                      x.isnumeric()]
    if len(selected_cards) == cards_select_limit and reward == max_reward:
        loop = False

    # Q_list = Q[current_state_index].tolist()
    # Q_list = sorted(Q_list)[::-1]
    #
    # # - refer the Q table to choose the most appropriate action for the state
    # action = Q[current_state_index].index(Q_list.pop())
    #
    # possible_states = Q[current_state_index][action]
    # if len(possible_states) == 0:  # handles the case when there are no states for a given action
    #     continue
    #
    # # - From the dictionary of possible_next_states, choose one state at random
    # # next_state = possible_states[random.randrange(len(possible_states))]
    # state_selected = None
    # reward_attained = 0
    # for i in range(0, len(possible_states)):
    #     state = possible_states[i]
    #     calculated_reward = calculate_reward(current_state, state)
    #     if calculated_reward < 0:  # handles case when reward for given state and action pair is penalising
    #         continue
    #
    #     reward_attained = calculate_reward
    #     state_selected = state
    #     break
    #
    # if not state_selected:  # handles the case when no state ,for given action, is rewarding
    #     continue
    #
    # reward += reward_attained
    #
    # current_state_index = env.state_to_index[state_selected]

# possible_states_in_ = env.possible_states_dict[current_state][action]

# - Calculate the total reward value achieved from attaining that state

# OUTPUT :: selected cards and order in which things were done
print("Reward:", reward)