import numpy as np
from custom_environment import WarEnv
import pickle
import matplotlib.pyplot as plt

alpha = 0.1  # Learning Rate
gamma = 0.95  # Discount Rate

total_episodes = 100000
max_steps = 10000

epsilon = 1
decay_rate = 1 / total_episodes

card_on_top = np.linspace(1, 14, 14)
player_highest = np.linspace(1, 14, 14)
player_lowest = np.linspace(1, 14, 14)
player_has_same_value = np.linspace(0, 1, 2)

print("Observation space ::", tuple([card_on_top, player_highest, player_lowest, player_has_same_value]))

def moving_average(data, window_size=30):
    # return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    averages = []
    start = 0
    end = 0
    for i in range(int(len(data) - window_size + 1)):
        end = start + window_size
        window = data[start: end]
        avg = sum(window) / window_size
        averages.append(avg)
        start = start + 1
    return averages


def create_environment():
    env = WarEnv()
    Q = np.zeros((len(card_on_top)+1, len(player_highest)+1, len(player_lowest)+1, len(player_has_same_value)+1, env.action_space.n))
    return env, Q


def choose_action(state, Q, env, epsilon):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
    return action


def update(state, state2, reward, action, action2, Q):
    predict = Q[state][action]
    target = reward + gamma * Q[state2][action2]
    Q[state][action] = Q[state][action] + alpha * (target - predict)
    return Q


def train(total_episodes, max_steps, env, Q, epsilon):
    data = {}
    win_history = []
    episodes_beating_game = 0
    for episode in range(total_episodes):
        data[episode] = 0
        t = 0
        state = env.reset()[0]
        state_card = np.digitize(state[0], card_on_top)
        state_player_highest_card = np.digitize(state[1], player_highest)
        state_player_lowest_card = np.digitize(state[2], player_lowest)
        state_player_has_same_value_card = np.digitize(state[3], player_has_same_value)
        state1 = (state_card, state_player_highest_card, state_player_lowest_card, state_player_has_same_value_card)

        action1 = choose_action(state1, Q, env, epsilon)

        # if episode % (total_episodes / 10) == 0:
        #     print("#### Episode:", episode, " :: action :", action1, ", Epsilon Value::", epsilon)
        while t < max_steps:

            # Visualizing the training
            # env.render()

            # Getting the next state
            state2, reward, done, trunc, info = env.step(action1)

            newstate_card = np.digitize(state2[0], card_on_top)
            newstate_player_highest_card = np.digitize(state2[1], player_highest)
            newstate_player_lowest_card = np.digitize(state2[2], player_lowest)
            newstate_player_has_same_value_card = np.digitize(state2[3], player_has_same_value)
            state2 = (newstate_card, newstate_player_highest_card, newstate_player_lowest_card, newstate_player_has_same_value_card)

            # Choosing the next action
            action2 = choose_action(state2, Q, env, epsilon)

            # Learning the Q-value
            Q = update(state1, state2, reward, action1, action2, Q)

            state1 = state2
            action1 = action2

            # Updating the respective values
            t += 1

            # If at the end of learning process
            if done:
                if reward > 0:
                    win_history.append(1)
                elif reward <= 0:  # loss
                    win_history.append(0)
                data[episode] = reward
                break

        # Decay epsilon
        epsilon = max(epsilon - decay_rate, 0)

    # saving the Q table in a file
    f = open("WARCardGame.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    reward_values = [data[ep] for ep in range(total_episodes) ]
    smoothed_rewards = moving_average(reward_values, window_size=10000)

    lists = sorted(data.items())  # sorted by key, return a list of tuples
    lists2 = sorted(data.values())  # sorted by values, return a list of values
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    # plt.plot(x, y)
    # plt.show()
    # plt.plot(lists2)
    # plt.show()

    plt.plot(smoothed_rewards)
    plt.show()

    smoothed_win_rate = [w * 100 for w in moving_average(win_history, window_size=10000)]
    plt.plot(smoothed_win_rate)
    plt.title("Win % (Moving Average)")
    # plt.xlabel("Episode")
    # plt.ylabel("Win Rate")
    plt.show()

    ## PLOTTING BOTH THE GRAPHS IN THE SAME IMAGE
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot(x, y)
    # ax1.set_title('Sharing Y axis')
    # ax2.plot(lists2)
    # plt.show()

    # print("Total Episodes where the agent aced the game:", episodes_beating_game)
    return Q


def play():
    env = WarEnv()

    # Read the Q table from the File
    f = open("WARCardGame.pkl", "rb")
    Q = pickle.load(f)
    f.close()

    won = 0
    tie = 0
    total = 100

    for i in range(total):
        t = 0
        reset_state = env.reset()[0]
        state_card = np.digitize(reset_state[0], card_on_top)
        state_player_highest_card = np.digitize(reset_state[1], player_highest)
        state_player_lowest_card = np.digitize(reset_state[2], player_lowest)
        state_player_has_same_value_card = np.digitize(reset_state[3], player_has_same_value)
        state = (state_card, state_player_highest_card, state_player_lowest_card, state_player_has_same_value_card)

        rewards = 0
        while t < max_steps:
            state2, reward, done, trunc, info = env.step(np.argmax(Q[state]))
            newstate_card = np.digitize(state2[0], card_on_top)
            newstate_player_highest_card = np.digitize(state2[1], player_highest)
            newstate_player_lowest_card = np.digitize(state2[2], player_lowest)
            newstate_player_has_same_value_card = np.digitize(state2[3], player_has_same_value)
            state = (newstate_card, newstate_player_highest_card, newstate_player_lowest_card,
                      newstate_player_has_same_value_card)
            t += 1
            rewards += reward

            if done:
                # print("Rewards ::", rewards, ", Info::", info, "state:", state2)
                won += 1
                break
            if t >= max_steps:
                # print("Max steps reached", ", Info::", info)
                tie += 1

        # print("STEPS ::: ", t)
    print("Rounds won:", (won/total)*100, ", Rounds Tied:", (tie/total)*100)


def main():
    env,Q = create_environment()
    Q = train(total_episodes, max_steps, env, Q, epsilon)
    # play()

main()