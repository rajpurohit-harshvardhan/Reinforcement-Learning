import gymnasium as gym
import numpy as np
import pickle
import tabulate
import math
import matplotlib.pyplot as plt

alpha = 0.9  # Learning Rate
gamma = 0.95  # Discount Rate

total_episodes = 100000
max_steps = 1000

epsilon = 1
decay_rate = 1 / total_episodes

# linspace for observation variables
lander_position_x = np.linspace(-0.8, 0.8, 17)
lander_position_y = np.linspace(-0.4, 1.6, 21)
lander_velocity_x = np.linspace(-1, 1, 21)
lander_velocity_y = np.linspace(-1.5, 0.5, 21)
lander_angle = np.linspace(-0.1, 0.1, 3)
lander_angular_velocity = np.linspace(-1.0, 1.0, 21)

observation_space = (lander_velocity_y, lander_angular_velocity)


# This function initializes the Gymnasium environment and creates a structure for Q table
def create_environment():
    env = gym.make('LunarLander-v2')

    # here Length+1 is done because in certain episodes the state values goes beyond the defined endpoints.
    Q = np.zeros((len(lander_velocity_y)+1, len(lander_angular_velocity)+1,  env.action_space.n))
    hit_table = np.zeros((len(lander_velocity_y)+1, len(lander_angular_velocity)+1))
    print("Linspace :: ", observation_space)
    return env, Q, hit_table


# This function helps with selection of the action based on the state and epsilon value
def choose_actions(state, env):
    # index representations
    # [0] = Distance in X coordinate
    # [1] = Distance in Y coordinate
    # [2] = Velocity in X direction
    # [3] = Velocity in Y direction
    # [4] = Angle
    # [5] = Angular Velocity

    action = 0

    # for distance in X coordinate
    # if state[0] < 0:
    #     action = 3
    # elif state[0] > 0:
    #     action = 1

    # for distance in Y coordinate
    # if state[1] < 0.3:
    #     action = 2
    # elif state[1] > 0.7:
    #     action = 0

    # for velocity in X direction
    # if state[2] < 0:
    #     action = 3
    # elif state[2] > 0:
    #     action = 1

    # for velocity in Y direction
    # if state[3] < 0:
    #     action = 2
    # elif state[3] > 0.2:
    #     action = 0

    # for angle
    # if state[4] < -0.18:
    #     action = 3
    # elif state[4] > 0.18:
    #     action = 1
    # else:
    #      action = 0

    # for angular velocity
    # if state[4] < -0.2:
    #     action = 3
    # elif state[4] > 0.2:
    #     action = 1
    # else:
    #     action = 2

    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:

        # Basically condition says that the acceptable angle for the lander is -10 to 10, if not that use side engines
        if -0.1 <= state[4] <= 0.1:
            # if the angular velocity is in CCW direction, turn on the right engine, 0.1 radia
            if state[5] < -0.1:
                action = 3
            # if the angular velocity is in CW direction, turn on the left engine
            elif state[5] > 0.1:
                action = 1
            else:
                # Once the angles are set and there are no angular rotations happening, We want to focus on the speed
                # vertical velocity, NEGATIVE = against gravity. the higher the -VE number, faster lander is falling
                if state[3] < -0.2:
                    action = 2
                # if vertical velocity is making the lander fly away rather than HOVER, do nothing
                elif state[3] > 0.3:
                    action = 0
                else:
                    # We focus on the velocity in the X direction to keep the lander upright
                    # 0.01 units OFF from the center of landing pad is still in the landing pad
                    if -0.01 < state[0] < 0.01:
                        # if velocity in X direction is higher NEGATIVE value (moving right), turn on left engine
                        if state[2] < -0.1:
                            action = 1
                        # if velocity in X direction is higher NEGATIVE value (moving left), turn on right engine
                        elif state[2] > 0.1:
                            action = 3
                        else:
                            # These next conditions are just to keep the lander hovering at a height just above ground
                            if state[1] < 0.2:
                                action = 2
                            elif state[1] > 0.5:
                                action = 0
                    elif state[0] < -0.01:
                        action = 1
                    elif state[0] > 0.01:
                        action = 3

        if state[4] < -0.1:  # -10 degrees
            action = 3
        elif state[4] > 0.1:  # 10 degrees
            action = 1
        # else:

    return action


# This function is responsible for updating the Q-value in the Q table for the given State-Action pair
def update(state1, state2, reward, action, action2, Q):
    predict = Q[state1][action]
    target = reward + gamma * Q[state2][action2]
    Q[state1][action] = Q[state1][action] + alpha * (target - predict)
    return Q


# This function is used for binning values in the defined bins/categories in the start using np.linspace
def digitize_states(value, digitize_space):
    return np.digitize(value, digitize_space, True)


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
def train(total_episodes, max_steps, env, Q, epsilon, hit_table):
    data = {}
    for episode in range(total_episodes):
        data[episode] = 0
        rewards = 0
        t = 0
        state1 = env.reset()[0]  # resetting the environment
        digitized_state_1_a = digitize_states(state1[3], lander_velocity_y)  # binning state value
        digitized_state_1_b = digitize_states(state1[5], lander_angular_velocity)  # binning state value
        digitized_state_1 = (digitized_state_1_a, digitized_state_1_b)
        hit_table[digitized_state_1_a][digitized_state_1_b] += 1  # recording a HIT

        action1 = choose_actions(state1, env)  # choosing an action based on the state

        if episode % (total_episodes/10) == 0:
            print("#### Episode:", episode, " :: action :", action1, ", Epsilon Value::", epsilon)


        episode_finished = False
        while not episode_finished:
            state2, reward, done, trunc, info = env.step(action1)  # Taking a step using environment

            action2 = choose_actions(state2, env)   # choosing an action based on the state

            digitized_state_2_a = digitize_states(state2[3], lander_velocity_y)  # binning state value
            digitized_state_2_b = digitize_states(state2[5], lander_angular_velocity)  # binning state value
            digitized_state_2 = (digitized_state_2_a, digitized_state_2_b)
            hit_table[digitized_state_2_a][digitized_state_2_b] += 1  # recording a HIT


            # if digitized_state_2 == 0:
            #     print("Episode :: ", episode, ", state_value :", state2[3])

            reward = reward_action(state1, state2)

            Q = update(digitized_state_1, digitized_state_2, reward, action1, action2, Q)  # updating the Q Value

            action1 = action2  # using the action deduced earlier for taking the next step.
            t+=1
            rewards += reward
            if done or trunc:
                data[episode] = rewards
                rewards = 0
                episode_finished = True
                break

            if t>max_steps:
                print("Exceeded max steps limit")
                episode_finished = True
                break

        epsilon = max(epsilon - decay_rate, 0)

    # saving the Q table in a file
    f = open("lunar_lander_conditions.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    lists = sorted(data.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    # plt.show()

    return Q, hit_table


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
            # if action2 == 0:
            #     print("doing nothing.", Q[digitized_state_2], digitized_state_2)

            action1 = action2  # using the action deduced earlier for taking the next step.
            t += 1

            if done:
                episode_finished = True
                break

            if t>max_steps:
                print("Exceeded max steps limit")
                episode_finished = True
                break


def display_table(hit_table):
    table = []
    velocity_y = np.append(lander_velocity_y, math.inf)
    angular_velocity = np.append(lander_angular_velocity, math.inf)

    for i in range(len(velocity_y)):
        record = [math.log(item, 10) if item > 0 else item for item in hit_table[i]]
        record = np.append(velocity_y[i], record)
        table.append(record)

    headers = np.append("Velocity in Y direction", angular_velocity)

    # saving the hit table in a file
    f = open("lunar_lander_conditions_hit_table.txt", "wb")
    pickle.dump(tabulate.tabulate(table, headers), f)
    f.close()

def main():
    env, Q, hit_table = create_environment()
    Q, hit_table = train(total_episodes, max_steps, env, Q, epsilon, hit_table)
    display_table(hit_table)
    # print(Q)
    play(Q)


main()
