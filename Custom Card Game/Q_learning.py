import numpy as np
from custom_environment import CardPickingEnv
import tabulate

alpha = 0.85
epsilon = 0.5
gamma = 0.95


def create_environment(cards_in_deck, cards_select_limit):
    env = CardPickingEnv(cards_in_deck, cards_select_limit)
    Q = np.zeros((env.observation_space.shape[0], env.action_space.n))
    return env, Q


def choose_action(state, Q, env):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def update(state, state2, reward, action, Q):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2])
    Q[state, action] = Q[state, action] + alpha * (target - predict)
    return Q


def train(total_episodes, max_steps, env, Q):
    for episode in range(total_episodes):
        t = 0
        state1 = env.reset()
        action1 = choose_action(state1, Q, env)
        reward = 0
        # print("Episode ::", episode)
        while t < max_steps:
            # Visualizing the training
            env.render()

            # Getting the next state
            state2, reward, done, info = env.step(action1)

            # Learning the Q-value
            Q = update(state1, state2, reward, action1, Q)

            state1 = state2

            # Updating the respective values
            t += 1
            # reward += 1

            # If at the end of learning process
            if done:
                break
    return Q


def display_Q_matrix(Q):
    print(Q)


def display_Q_table(env, Q):
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