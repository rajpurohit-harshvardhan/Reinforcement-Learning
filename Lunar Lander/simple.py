import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

alpha = 0.9  # Learning Rate
gamma = 0.95  # Discount Rate

total_episodes = 10000
max_steps = 1000

epsilon = 1
decay_rate = 1 / total_episodes

# linspace for observation variables
lander_position_x = np.linspace(-0.8, 0.8, 17)
lander_position_y = np.linspace(-0.4, 1.6, 21)
lander_velocity_x = np.linspace(-1, 1, 21)
lander_velocity_y = np.linspace(-0.5, 0.5, 21)
lander_angle = np.linspace(-0.1, 0.1, 3)
lander_angular_velocity = np.linspace(-0.2, 0.2, 21)

observation_space = (lander_velocity_y, lander_angular_velocity)


# This function initializes the Gymnasium environment and creates a structure for Q table
def create_environment():
    env = gym.make('LunarLander-v2')

    # here Length+1 is done because in certain episodes the state values goes beyond the defined endpoints.
    Q = np.zeros((len(lander_velocity_y)+1, len(lander_angular_velocity)+1,  env.action_space.n))  # Init Q table
    print("Linspace :: ", observation_space)
    return env, Q


# This function helps with selection of the action based on the state and epsilon value
def choose_actions(state, env, Q):
    # index representations
    # [0] = Distance in X coordinate
    # [1] = Distance in Y coordinate
    # [2] = Velocity in X direction
    # [3] = Velocity in Y direction
    # [4] = Angle
    # [5] = Angular Velocity

    action = 0

    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        x_distance = state[0]
        y_distance = state[1]
        x_velocity = state[2]
        y_velocity = state[3]
        angle = state[4]
        angular_velocity = state[5]

        if -0.1 <= angle <= 0.1:
            # if the angular velocity is in CCW direction, turn on the right engine
            if angular_velocity < -0.05:  # CHANGED FROM 0.1
                action = 1
            # if the angular velocity is in CW direction, turn on the left engine
            elif angular_velocity > 0.05:  # CHANGED FROM 0.1
                action = 3
            else:
               action = 0
        if angle < -0.1:  # -10 degrees
            action = 1
        elif angle > 0.1:  # 10 degrees
            action = 3
        # action = np.argmax(Q[state])

    return action


# This function is responsible for updating the Q-value in the Q table for the given State-Action pair
def update(state1, state2, reward, action, action2, Q):
    predict = Q[state1][action]
    target = reward + gamma * Q[state2][action2]  # discounting the next possible Q-value of state-action pair
    Q[state1][action] = Q[state1][action] + alpha * (target - predict)  # incorporating learnings with the current Q-value
    return Q


# This function is used for binning values in the defined bins/categories in the start using np.linspace
def digitize_states(value, digitize_space):
    return np.digitize(value, digitize_space, True)
    # True here means it will include the RHS value in the bin so a bin of 0.5 to 0.6 = [0.5, 0.6] and NOT [0.5,0.6)


# This function is used for the calculation of reward values based on the state.
def reward_action(state1, state2):
    total_reward = 0

    x_distance = state2[0]
    y_distance = state2[1]
    x_velocity = state2[2]
    y_velocity = state2[3]
    angle = state2[4]
    angular_velocity = state2[5]

    if x_distance < 0:
        total_reward += x_distance
    elif x_distance > 0:
        total_reward += -x_distance
    else: total_reward += 1

    if y_distance > 0.4:
        total_reward += -y_distance
    elif y_distance < 0.1:
        total_reward += 1

    if x_velocity < 0:
        total_reward += x_velocity * 10
    elif x_velocity > 0:
        total_reward += -x_velocity * 10
    else: total_reward += 10

    if y_velocity > 0.2:
        total_reward += -y_velocity * 10
    elif y_velocity < -0.1:
        total_reward += y_velocity * 10
    else:
        total_reward += 10

    if angle > 0.1:
        total_reward += -angle * 10
    elif angle < -0.1:
        total_reward += angle * 10
    else:
        total_reward += 10

    if angular_velocity > 0.2:
        total_reward += -angular_velocity
    elif angular_velocity < -0.2:
        total_reward += angular_velocity
    else:
        total_reward += 1

    if state2[6] == 1 and state2[7] == 1:
        if y_velocity < -0.1:
            total_reward += -1000
        else:
            total_reward += 1000

    # elif state2[6] == 1 or state1[6] == 1:
    #     if y_velocity < -0.1:
    #         total_reward += -1000
    #     else:
    #         total_reward += -500

    return total_reward


# This function is responsible for the training of the agent
def train(total_episodes, max_steps, env, Q, epsilon):
    data = {}
    for episode in range(total_episodes):
        data[episode] = 0
        rewards = 0
        t = 0
        state1 = env.reset()[0]  # resetting the environment, receives a list of size 7 in return
        digitized_state_1_a = digitize_states(state1[3], lander_velocity_y)  # Binning state value
        digitized_state_1_b = digitize_states(state1[5], lander_angular_velocity)  # binning state value
        digitized_state_1 = (digitized_state_1_a, digitized_state_1_b)

        action1 = choose_actions(state1, env, Q)  # choosing an action based on the state

        if episode % (total_episodes/10) == 0:
            print("#### Episode:", episode, " :: action :", action1, ", Epsilon Value::", epsilon)


        episode_finished = False
        while not episode_finished:
            state2, reward, done, trunc, info = env.step(action1)  # Taking a step using environment

            action2 = choose_actions(state2, env, Q)   # choosing an action based on the state

            digitized_state_2_a = digitize_states(state2[3], lander_velocity_y)  # binning state value
            digitized_state_2_b = digitize_states(state2[5], lander_angular_velocity)  # binning state value
            digitized_state_2 = (digitized_state_2_a, digitized_state_2_b)

            reward = reward_action(state1, state2)  # find out the reward based on the current state

            Q = update(digitized_state_1, digitized_state_2, reward, action1, action2, Q)  # update the Q Value

            action1 = action2  # using the action deduced earlier for taking the next step.

            t+=1  # used for terminating the training if maximum steps are achieved
            rewards += reward  # used for calculating total reward throughout the episode

            # this condition truncates or ends the training if the environment ends it
            if done or trunc:
                data[episode] = rewards
                rewards = 0
                episode_finished = True
                break

            # this condition terminates the training if maximum steps are taken
            if t > max_steps:
                print("Exceeded max steps limit")
                episode_finished = True
                break

        epsilon = max(epsilon - decay_rate, 0)  # reducing the epsilon value by the decay rate

    # saving the Q table in a file
    f = open("lunar_lander_conditions.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    lists = sorted(data.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    # plt.show()  # used to show the graph for number-of-steps X episodes

    return Q


# this function is responsible for rendering the agent playing the game using the Q table we computed in the training
def play(Q):
    env = gym.make('LunarLander-v2', render_mode='human')

    # Read the Q table from the File
    f = open("lunar_lander_conditions.pkl", "rb")
    Q = pickle.load(f)
    f.close()

    for i in range(30):
        t = 0
        state1 = env.reset()[0]
        digitized_state_1_a = digitize_states(state1[3], lander_velocity_y)  # binning state value
        digitized_state_1_b = digitize_states(state1[5], lander_angular_velocity)  # binning state value
        digitized_state_1 = (digitized_state_1_a, digitized_state_1_b)
        print("STARTING STATE :::: ", state1, digitized_state_1)

        action1 = np.argmax(Q[digitized_state_1])  # since Q table stores the values with digitized state

        episode_finished = False
        while not episode_finished:
            state2, reward, done, trunc, info = env.step(action1)  # Taking a step using environment

            digitized_state_2_a = digitize_states(state2[3], lander_velocity_y)  # binning state value
            digitized_state_2_b = digitize_states(state2[5], lander_angular_velocity)  # binning state value
            digitized_state_2 = (digitized_state_2_a, digitized_state_2_b)

            action2 = np.argmax(Q[digitized_state_2]) # since Q table stores the values with digitized state

            action1 = action2  # using the action deduced earlier for taking the next step.
            t += 1  # used for terminating the training if maximum steps are achieved

            # this condition truncates or ends the training if the environment ends it
            if done:
                episode_finished = True
                break

            # this condition terminates the training if maximum steps are taken
            if t>max_steps:
                print("Exceeded max steps limit")
                episode_finished = True
                break


def main():
    env, Q = create_environment()
    Q = train(total_episodes, max_steps, env, Q, epsilon)
    play(Q)


main()
