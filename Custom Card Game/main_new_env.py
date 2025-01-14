import gymnasium as gym
import numpy as np
from custom_environment_2  import AccumulateHighestCardsEnv
import pickle
import matplotlib.pyplot as plt

alpha = 0.1  # Learning Rate
gamma = 0.95  # Discount Rate

total_episodes = 100000
max_steps = 1000

epsilon = 0.9
epsilon_min = 0.1
epsilon_decay = 0.995

player_score = np.linspace(1, 52, 52)
card_on_top = np.linspace(1, 13, 13)

print("Observation space ::", tuple([player_score, card_on_top]))


def create_environment():
    env = AccumulateHighestCardsEnv()
    Q = np.zeros((len(player_score)+1, len(card_on_top)+1, env.action_space.n))
    return env, Q


def choose_action(state, Q, env):
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
    episodes_beating_game = 0
    for episode in range(total_episodes):
        data[episode] = 0
        t = 0
        state = env.reset()[0]
        state_player = np.digitize(state[0], player_score )
        state_card = np.digitize(state[1], card_on_top)
        state1 = (state_player, state_card)

        action1 = choose_action(state1, Q, env)

        if episode % (total_episodes / 10) == 0:
            print("#### Episode:", episode, " :: action :", action1, ", Epsilon Value::", epsilon)
        while t < max_steps:

            # Visualizing the training
            # env.render()

            # Getting the next state
            state2, reward, done, trunc, info = env.step(action1)
            newstate_player = np.digitize(state2[0], player_score)
            newstate_card = np.digitize(state2[1], card_on_top)
            state2 = (newstate_player, newstate_card)

            # Choosing the next action
            action2 = choose_action(state2, Q, env)

            # Learning the Q-value
            Q = update(state1, state2, reward, action1, action2, Q)

            state1 = state2
            action1 = action2

            # Updating the respective values
            t += 1

            # If at the end of learning process
            if done:
                data[episode] = reward
                break

            # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    # saving the Q table in a file
    f = open("CustomCardGame.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    lists = sorted(data.items())  # sorted by key, return a list of tuples
    lists2 = sorted(data.values())  # sorted by values, return a list of values
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    # plt.plot(x, y)
    # plt.show()
    plt.plot(lists2)
    # plt.show()

    ## PLOTTING BOTH THE GRAPHS IN THE SAME IMAGE
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot(x, y)
    # ax1.set_title('Sharing Y axis')
    # ax2.plot(lists2)
    # plt.show()

    # print("Total Episodes where the agent aced the game:", episodes_beating_game)
    return Q


def play():
    env = AccumulateHighestCardsEnv()

    # Read the Q table from the File
    f = open("CustomCardGame.pkl", "rb")
    Q = pickle.load(f)
    f.close()

    for i in range(20):
        t = 0
        reset_state = env.reset()[0]
        state_player = np.digitize(reset_state[0], player_score)
        state_card = np.digitize(reset_state[1], card_on_top)
        state = (state_player, state_card)

        rewards = 0
        while t < max_steps:
            state2, reward, done, trunc, info = env.step(np.argmax(Q[state]))
            newstate_player = np.digitize(state2[0], player_score)
            newstate_card = np.digitize(state2[1], card_on_top)
            state = (newstate_player, newstate_card)

            t += 1
            rewards += reward

            if done:
                print("Rewards ::", rewards, ", Info::", info)
                print(newstate_player, newstate_card)
                break

        # print("STEPS ::: ", t)

def main():
    env,Q = create_environment()
    # Q = train(total_episodes, max_steps, env, Q, epsilon)
    play()

main()