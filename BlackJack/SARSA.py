import gymnasium as gym
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

alpha = 0.001  # Learning Rate
gamma = 0.001  # Discount RateA

total_episodes = 100000
max_steps = 1000

epsilon = 1
decay_rate = 1 / total_episodes

player_score = np.linspace(4, 30, 27)
dealer_score = np.linspace(2, 11, 10)
usable_ace = np.linspace(0, 1, 2)

print("Observation space ::", tuple([player_score, dealer_score, usable_ace]))


def moving_average(data, window_size):
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
    env = gym.make('Blackjack-v1')
    Q = np.zeros((len(player_score)+1, len(dealer_score)+1, len(usable_ace), env.action_space.n))
    return env, Q


def choose_action(state, Q, env, epsilon):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])

        # player_card_value = state[0]
        # dealer_card_value = state[1]
        #
        # if player_card_value >= 17:
        #     action = 0
        # elif 12 <= player_card_value <= 16:
        #     if player_card_value == 16 and 9 <= dealer_card_value <= 11:
        #         action = 0
        #     elif player_card_value == 15 and dealer_card_value == 10:
        #         action = 0
        #     elif player_card_value == 12 and 2 <= dealer_card_value <= 3:
        #         action = 1
        #     elif 7 <= dealer_card_value <= 11:
        #         action = 1
        #     else:
        #         action = 0
        # else:
        #     action = 1
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
        state_player = np.digitize(state[0], player_score, True)
        state_dealer = np.digitize(state[1], dealer_score, True)
        state_ace = np.digitize(state[2], usable_ace, True)
        state1 = (state_player, state_dealer, state_ace)

        action1 = choose_action(state1, Q, env, epsilon)
        # if episode % (total_episodes / 10) == 0:
        #     print("#### Episode:", episode, " :: action :", action1, ", Epsilon Value::", epsilon)
        # print("Episode ::", episode)
        while t < max_steps:
            # Visualizing the training
            # env.render()

            # Getting the next state
            state2, reward, done, trunc, info = env.step(action1)
            newstate_player = np.digitize(state2[0], player_score, True)
            newstate_dealer = np.digitize(state2[1], dealer_score, True)
            newstate_ace = np.digitize(state2[2], usable_ace, True)
            state2 = (newstate_player, newstate_dealer, newstate_ace)


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
                # if reward == 0:
                #     reward -= 0.01
                # elif reward >0:
                #     reward += 0.01

                if reward == 1:
                    win_history.append(1)
                else:  # loss
                    win_history.append(0)

                data[episode] = reward
                # if t > 500:
                #     episodes_beating_game += 1
                break

        epsilon = max(epsilon - decay_rate, 0)

    # saving the Q table in a file
    f = open("blackjack.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    reward_values = [data[ep] for ep in range(total_episodes)]
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
    # plt.title("Win % (Moving Average)")
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
    wins = 0
    losses = 0
    ties = 0
    env = gym.make('Blackjack-v1')

    # Read the Q table from the File
    f = open("blackjack.pkl", "rb")
    Q = pickle.load(f)
    f.close()

    for i in range(10000):
        t = 0
        reset_state = env.reset()[0]
        state_player = np.digitize(reset_state[0], player_score, True)
        state_dealer = np.digitize(reset_state[1], dealer_score, True)
        state_ace = np.digitize(reset_state[2], usable_ace, True)
        state = (state_player, state_dealer, state_ace)
        rewards = 0
        while t < max_steps:
            state2, reward, done, trunc, info = env.step(np.argmax(Q[state]))
            newstate_player = np.digitize(state2[0], player_score, True)
            newstate_dealer = np.digitize(state2[1], dealer_score, True)
            newstate_ace = np.digitize(state2[2], usable_ace, True)
            state = (newstate_player, newstate_dealer, newstate_ace)

            t += 1
            rewards += reward

            if done:
                if reward >= 1:
                    wins += 1
                elif reward == -1:
                    losses += 1
                elif reward == 0:
                    ties += 1

                # print("Rewards ::", rewards)
                break

        # print("STEPS ::: ", t)
    print("Wins::", wins/100, ", Losses:", losses/100, ", Ties:", ties/100)

def main():
    env,Q = create_environment()
    # print(env, Q)
    Q = train(total_episodes, max_steps, env, Q, epsilon)
    play()

main()