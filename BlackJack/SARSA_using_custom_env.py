import gymnasium as gym
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from custom_environment import BlackJackEnv
import tabulate

alpha = 0.001  # Learning Rate
gamma = 0.001  # Discount Rate

total_episodes = 100000
max_steps = 1000

epsilon = 1
decay_rate = 1 / total_episodes

player_score = np.linspace(4, 30, 27)
dealer_score = np.linspace(2, 11, 10)
usable_ace = np.linspace(0, 1, 2)

print("Observation space ::", tuple([player_score, dealer_score, usable_ace]))


def create_environment():
    env = BlackJackEnv(has_infinite_deck=True)
    # env = gym.make('Blackjack-v1')
    Q = np.zeros((len(player_score)+1, len(dealer_score)+1, len(usable_ace), env.action_space.n))
    hit_table = np.zeros((len(player_score), len(dealer_score)))
    return env, Q, hit_table


def choose_action(state, Q, env):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        # action = np.argmax(Q[state])

        # action :: 0 = STAND / 1 = HIT
        action = 0
        player_card_value = state[0]
        dealer_card_value = state[1]

        if player_card_value >= 17:
            action = 0
        elif 12 <= player_card_value <= 16:
            if player_card_value == 16 and 9 <= dealer_card_value <= 11:
                action = 0
            elif player_card_value == 15 and dealer_card_value == 10:
                action = 0
            elif player_card_value == 12 and 2 <= dealer_card_value <= 3:
                action = 1
            elif 7 <= dealer_card_value <= 11:
                action = 1
            else:
                action = 0
        else:
            action = 1

    return action


def update(state, state2, reward, action, action2, Q):
    predict = Q[state][action]
    target = reward + gamma * Q[state2][action2]
    Q[state][action] = Q[state][action] + alpha * (target - predict)
    return Q


def train(total_episodes, max_steps, env, Q, epsilon, hit_table):
    data = {}
    episodes_beating_game = 0
    for episode in range(total_episodes):
        data[episode] = 0
        t = 0
        state = env.reset()[0]
        state_player = np.digitize(state[0], player_score, True)
        state_dealer = np.digitize(state[1], dealer_score, True)
        state_ace = np.digitize(state[2], usable_ace, True)
        state1 = (state_player, state_dealer, state_ace)

        # hit_table[state_player][state_dealer] += 1

        action1 = choose_action(state1, Q, env)
        # print("Episode ::", episode)
        while t < max_steps:
            # Visualizing the training
            # env.render()

            # Getting the next state
            state2, reward, done, trunc, info = env.step(action1)
            newstate_player = np.digitize(state2[0], player_score, True)
            newstate_dealer = np.digitize(state2[1], dealer_score, True)
            newstate_ace = np.digitize(state2[2], usable_ace, True)

            # if state2[1] == 2:
            #     print("Achieved state ::", newstate_dealer)

            state2 = (newstate_player, newstate_dealer, newstate_ace)

            # hit_table[newstate_player][newstate_dealer] += 1

            # Modify reward based on current state and action
            # if action1 == 1 and state[0] >= 17:  # Penalty for hitting when close to busting
            #     reward -= 0.5
            # elif action1 == 0 and 18 <= state[0] <= 21:  # Reward for standing with a strong hand
            #     reward += 0.5

            # Choosing the next action
            action2 = choose_action(state2, Q, env)

            # Learning the Q-value
            Q = update(state1, state2, reward, action1, action2, Q)

            state1 = state2
            action1 = action2
            t += 1

            if done:
                data[episode] = reward
                break

        # epsilon = max(epsilon - decay_rate, 0)
        epsilon = max(epsilon - decay_rate, 0)

    # saving the Q table in a file
    f = open("blackjack_custom.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    lists = sorted(data.items())  # sorted by key, return a list of tuples
    lists2 = sorted(data.values())  # sorted by values, return a list of values
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.show()
    plt.plot(lists2)
    plt.show()

    ## PLOTTING BOTH THE GRAPHS IN THE SAME IMAGE
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot(x, y)
    # ax1.set_title('Sharing Y axis')
    # ax2.plot(lists2)
    # plt.show()

    # print("Total Episodes where the agent aced the game:", episodes_beating_game)
    return Q, hit_table


def play():
    wins = 0
    losses = 0
    ties = 0
    # env = gym.make('Blackjack-v1')
    env = BlackJackEnv(has_infinite_deck=True)

    # Read the Q table from the File
    f = open("blackjack_custom.pkl", "rb")
    Q = pickle.load(f)
    f.close()

    for i in range(100000):
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
                # print("External Information ::", trunc)
                if reward >= 1:
                    wins += 1
                elif reward == -1:
                    losses += 1
                elif reward == 0:
                    ties += 1

                # print("Rewards ::", rewards)
                break

        # print("STEPS ::: ", t)
    print("Wins::", wins, ", Losses:", losses, ", Ties:", ties)


def display_table(hit_table):
    table = []

    for i in range(len(player_score)):
        record = [round(math.log(item, 10), 1) if item > 0 else item for item in hit_table[i]]
        record = np.append(player_score[i], record)
        table.append(record)

    headers = np.append("Dealer Score", dealer_score)
    print(tabulate.tabulate(table, headers))


def main():
    env, Q, hit_table = create_environment()
    # print(env, Q)
    Q, hit_table = train(total_episodes, max_steps, env, Q, epsilon, hit_table)
    # display_table(hit_table)
    # play()


main()
