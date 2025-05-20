import gymnasium as gym
import numpy as np
import pickle
import tabulate
import math
import matplotlib.pyplot as plt

alpha = 0.1  # Learning Rate
gamma = 0.9  # Discount Rate

total_episodes = 150000
max_steps = 1000

min_epsilon = 0.001
max_epsilon = 1.0
decay_rate = 0.00002
epsilon = 1

# linspace for observation variables
lander_position_x = np.linspace(-0.8, 0.8, 17)
lander_position_y = np.linspace(-0.4, 1.6, 21)
lander_velocity_x = np.linspace(-1, 1, 21)
lander_velocity_y = np.linspace(-1.0, 0.6, 21)
lander_angle = np.linspace(-1.1, 1.1, 21)
lander_angular_velocity = np.linspace(-1.1, 1.1, 21)

observation_space = (lander_velocity_y, lander_angular_velocity)


# This function initializes the Gymnasium environment and creates a structure for Q table
def create_environment():
    env = gym.make('LunarLander-v3')

    # here Length+1 is done because in certain episodes the state values goes beyond the defined endpoints.
    Q = np.zeros((len(lander_velocity_y) + 1, len(lander_angular_velocity) + 1, env.action_space.n))
    hit_table = np.zeros((len(lander_velocity_y) + 1, len(lander_angular_velocity) + 1))
    Q_x = np.zeros((len(lander_velocity_x) + 1, env.action_space.n))
    print("Linspace :: ", observation_space)
    return env, Q, hit_table, Q_x


# This function helps with selection of the action based on the state and epsilon value
def choose_actions(state, env, Q):
    # index representations
    # [0] = Distance in X coordinate
    # [1] = Distance in Y coordinate
    # [2] = Velocity in X direction
    # [3] = Velocity in Y direction
    # [4] = Angle
    # [5] = Angular Velocity
    x_distance = state[0]
    y_distance = state[1]
    x_velocity = state[2]
    y_velocity = state[3]
    angle = state[4]
    angular_velocity = state[5]

    action = 0

    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:

        # Basically condition says that the acceptable angle for the lander is -10 to 10, if not that use side engines
        if -0.05 <= angle <= 0.05:
            # if the angular velocity is in CCW direction, turn on the right engine
            if angular_velocity < -0.05:  # CHANGED FROM 0.1
                action = 1
            # if the angular velocity is in CW direction, turn on the left engine
            elif angular_velocity > 0.05:  # CHANGED FROM 0.1
                action = 3
            else:
                # Once the angles are set and there are no angular rotations happening, We want to focus on the speed
                # vertical velocity, NEGATIVE = against gravity. the higher the -VE number, faster lander is falling
                if y_velocity < -0.1:
                    action = 2
                # if vertical velocity is making the lander fly away rather than HOVER, do nothing
                elif y_velocity > 0.05:
                    action = 0
                else:
                    # We focus on the velocity in the X direction to keep the lander upright
                    # 0.01 units OFF from the center of landing pad is still in the landing pad
                    if -0.01 <= x_distance <= 0.01:
                        # if velocity in X direction is higher NEGATIVE value (moving right), turn on left engine
                        if x_velocity < 0:
                            action = 1
                        # if velocity in X direction is higher NEGATIVE value (moving left), turn on right engine
                        elif x_velocity > 0.01:
                            action = 3
                        else:
                            # These next conditions are just to keep the lander hovering at a height just above ground
                            if y_distance <= 0.4:
                                action = 2
                            elif y_distance > 0.5:
                                action = 0
                    elif x_distance < -0.15:
                        action = 1
                    elif x_distance > 0.15:
                        action = 3
                    else:
                        print("Distance in X direction=", x_distance)

        if angle < -0.1:  # -10 degrees
            action = 1
        elif angle > 0.1:  # 10 degrees
            action = 3
        # else:
        action = np.argmax(Q[state])
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
def reward_action(reward, state2):
    total_reward = reward

    x_distance = state2[0]
    y_distance = state2[1]
    x_velocity = state2[2]
    y_velocity = state2[3]
    angle = state2[4]
    angular_velocity = state2[5]

    # if x_distance < 0:
    #     total_reward += x_distance * 5
    # elif x_distance > 0:
    #     total_reward += -x_distance * 5
    # else:
    #     total_reward += x_distance
    #
    # if y_distance > 0.1:
    #     total_reward += -y_distance
    # elif y_distance <= 0.1:
    #     total_reward += y_distance
    #
    # if x_velocity < 0:
    #     total_reward += x_velocity + 5
    # elif x_velocity > 0:
    #     total_reward += -x_velocity - 5
    # else:
    #     total_reward += 1
    #
    # if y_velocity > 0:
    #     total_reward += -y_velocity
    # elif y_velocity < -0.1:
    #     total_reward += y_velocity
    # else:
    #     total_reward += y_velocity
    #
    # if angle > 0.05:
    #     total_reward -= angle + 5
    # elif angle < -0.05:
    #     total_reward += angle - 5
    # else:
    #     total_reward += angle
    #
    # if angular_velocity > 0.1:
    #     total_reward -= angular_velocity + 5
    # elif angular_velocity < -0.1:
    #     total_reward += angular_velocity - 5
    # else:
    #     total_reward += angular_velocity
    #
    # if state2[6] == 1 and state2[7] == 1:
    #     if -0.1 <= round(y_velocity, 2) < 0.1 and -0.01 < round(angle, 2) < 0.01 and -0.1 < round(x_velocity, 2) < 0.1:
    #         # print("SUccessfull Landing", round(y_velocity, 2), -0.1 <= y_velocity < 0.1, round(angle, 2), (-0.01 < angle < 0.01),  round(x_velocity, 2), (-0.1 < x_velocity < 0.1))
    #         total_reward += 1000
    #     else:
    #         total_reward -= 100

    # elif state2[6] == 1 or state1[6] == 1:
    #     if y_velocity < -0.1:
    #         total_reward += -100
    #     else:
    #         total_reward += -100

    # High altitude adjustments
    if y_distance > 0.5:
        total_reward += -abs(x_velocity) * 5  # Light penalty for x_velocity

    # Mid altitude adjustments
    elif 0.1 < y_distance <= 0.5:
        total_reward += -abs(x_velocity) * 5  # Stronger penalty for horizontal drift
        total_reward += -abs(angle) * 7  # Penalty for any tilt

    # Low altitude adjustments
    else:
        total_reward += -abs(
            y_velocity) * 20 if y_velocity < -0.1 else y_velocity * 10  # High penalty for fast descent, reward slow descent
        total_reward += -abs(x_velocity) * 10 if abs(x_distance) > 0.05 else -abs(
            x_velocity) * 20  # Strong incentive to stay near center

    # Stabilize angle and angular velocity
    total_reward += -abs(angle) * 10
    total_reward += -abs(angular_velocity) * 5

    # Reward for stable near-landing conditions
    if -0.1 <= x_distance <= 0.1 and -0.05 <= angle <= 0.05 and -0.1 < y_velocity < 0.1:
        total_reward += 500  # Strong incentive for stable, center-aligned descent

    # Large reward for perfect landing
    if -0.05 < x_distance < 0.05 and -0.1 < y_velocity < 0.1 and -0.05 < angle < 0.05:
        total_reward += 1000

    return total_reward


def reward_action_formula(reward, state2):
    total_reward = reward

    x_distance = state2[0]
    y_distance = state2[1]
    x_velocity = state2[2]
    y_velocity = state2[3]
    angle = state2[4]
    angular_velocity = state2[5]

    total_reward += -100 * math.sqrt(math.pow(x_distance, 2) + math.pow(y_distance, 2))
    total_reward += -100 * math.sqrt(math.pow(x_velocity, 2) + math.pow(y_velocity, 2))
    total_reward += -100 * math.fabs(angular_velocity)
    total_reward += 10 * state2[6]
    total_reward += 10 * state2[7]

    if state2[6] == 1 and state2[7] == 1:
        if -0.1 <= round(y_velocity, 2) < 0.1 and -0.01 < round(angle, 2) < 0.01 and -0.1 < round(x_velocity, 2) < 0.1:
            # print("SUccessfull Landing", round(y_velocity, 2), -0.1 <= y_velocity < 0.1, round(angle, 2), (-0.01 < angle < 0.01),  round(x_velocity, 2), (-0.1 < x_velocity < 0.1))
            total_reward += 100
        else:
            total_reward -= 100

    # elif state2[6] == 1 or state1[6] == 1:
    #     if y_velocity < -0.1:
    #         total_reward += -100
    #     else:
    #         total_reward += -100

    return total_reward


# This function is responsible for the training of the agent
def train(total_episodes, max_steps, env, Q, epsilon, hit_table, Q_x):
    data = {}
    for episode in range(total_episodes):
        data[episode] = 0
        rewards = 0
        t = 0
        state1 = env.reset()[0]  # resetting the environment
        digitized_state_1_a = digitize_states(math.tanh(state1[3]), lander_velocity_y)  # binning state value
        digitized_state_1_b = digitize_states(math.tanh(state1[5]), lander_angular_velocity)  # binning state value
        digitized_state_1 = (digitized_state_1_a, digitized_state_1_b)
        hit_table[digitized_state_1_a][digitized_state_1_b] += 1  # recording a HIT

        digitized_state_1_x = (digitize_states(math.tanh(state1[2]), lander_velocity_x))  # binning state value

        action1 = choose_actions(state1, env, Q)  # choosing an action based on the state

        if episode % (total_episodes / 10) == 0:
            print("#### Episode:", episode, " :: action :", action1, ", Epsilon Value::", epsilon)

        episode_finished = False
        while not episode_finished:
            state2, reward, done, trunc, info = env.step(action1)  # Taking a step using environment

            action2 = choose_actions(state2, env, Q)  # choosing an action based on the state

            digitized_state_2_a = digitize_states(math.tanh(state2[3]), lander_velocity_y)  # binning state value
            digitized_state_2_b = digitize_states(math.tanh(state2[5]), lander_angular_velocity)  # binning state value
            digitized_state_2 = (digitized_state_2_a, digitized_state_2_b)
            hit_table[digitized_state_2_a][digitized_state_2_b] += 1  # recording a HIT

            digitized_state_2_x = (digitize_states(math.tanh(state2[2]), lander_velocity_x))  # binning state value

            # if digitized_state_2 == 0:
            #     print("Episode :: ", episode, ", state_value :", state2[3])

            # reward = reward_action(reward, state2)
            reward = reward_action_formula(reward, state2)

            Q = update(digitized_state_1, digitized_state_2, reward, action1, action2, Q)  # updating the Q Value
            Q_x = update(digitized_state_1_x, digitized_state_2_x, reward, action1, action2,
                         Q_x)  # updating the Q Value

            action1 = action2  # using the action deduced earlier for taking the next step.
            t += 1
            rewards += reward
            if done or trunc:
                data[episode] = rewards
                rewards = 0
                episode_finished = True
                break

            if t > max_steps:
                print("Exceeded max steps limit")
                episode_finished = True
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    # saving the Q table in a file
    f = open("lunar_lander_conditions_new_tanh.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    f = open("lunar_lander_conditions_new_tanh_QX.pkl", "wb")
    pickle.dump(Q_x, f)
    f.close()

    lists = sorted(data.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    # plt.show()

    return Q, hit_table, Q_x


# this function is responsible for rendering the agent playing the game using the Q table we computed in the training
def play():
    env = gym.make('LunarLander-v3')

    # Read the Q table from the File
    f = open("lunar_lander_conditions_new_tanh.pkl", "rb")
    Q = pickle.load(f)
    f.close()

    # Read the Q table from the File
    f = open("lunar_lander_conditions_new_tanh_QX.pkl", "rb")
    Q_x = pickle.load(f)
    f.close()

    for i in range(20):
        print('#' * 50)
        t = 0
        total_rewards = 0
        state1 = env.reset()[0]
        digitized_state_1_a = digitize_states(math.tanh(state1[3]), lander_velocity_y)  # binning state value
        digitized_state_1_b = digitize_states(math.tanh(state1[5]), lander_angular_velocity)  # binning state value
        digitized_state_1 = (digitized_state_1_a, digitized_state_1_b)
        # print("STARTING STATE :::: ", state1, digitized_state_1)

        digitized_state_1_x = (digitize_states(math.tanh(state1[2]), lander_velocity_x))  # binning state value

        actionQ = np.argmax(Q[digitized_state_1])  # since Q table stores the values with digitized state
        actionQ_x = np.argmax(Q_x[digitized_state_1_x])  # since Q table stores the values with digitized state

        if True:
            action1 = actionQ
        else:
            action1 = actionQ_x

        episode_finished = False
        counter = 0
        while not episode_finished:
            state2, reward, done, trunc, info = env.step(action1)  # Taking a step using environment

            digitized_state_2_a = digitize_states(math.tanh(state2[3]), lander_velocity_y)  # binning state value
            digitized_state_2_b = digitize_states(math.tanh(state2[5]), lander_angular_velocity)  # binning state value
            digitized_state_2 = (digitized_state_2_a, digitized_state_2_b)

            digitized_state_2_x = (digitize_states(math.tanh(state2[2]), lander_velocity_x))  # binning state value

            actionQ = np.argmax(Q[digitized_state_2])  # since Q table stores the values with digitized state
            actionQ_x = np.argmax(Q_x[digitized_state_2_x])  # since Q table stores the values with digitized state

            if True:
                action2 = actionQ
            else:
                action2 = actionQ_x

            # action2 = np.argmax(Q[digitized_state_2]) # since Q table stores the values with digitized state
            # if action2 == 0:
            #     print("doing nothing.", Q[digitized_state_2], digitized_state_2, state2)

            action1 = action2  # using the action deduced earlier for taking the next step.
            t += 1
            total_rewards += reward

            if done:
                episode_finished = True
                print("Episode finished with reward value ::: ", total_rewards)
                break

            if t > max_steps:
                print("Exceeded max steps limit")
                episode_finished = True
                break


def display_table(hit_table):
    #  VALUES IN TOP = lander_angle values
    #  Values going down = Velocity in Y direction

    table = []
    velocity_y = np.append(lander_velocity_y, math.inf)
    angular_velocity = np.append(lander_angle, math.inf)

    for i in range(len(velocity_y)):
        # record = [math.log(item, 10) if item > 0 else item for item in hit_table[i]]
        record = [item for item in hit_table[i]]
        record = np.append(velocity_y[i], record)
        table.append(record)

    headers = np.append("Velocity in Y direction", angular_velocity)

    # saving the hit table in a file
    f = open("lunar_lander_conditions_new_tanh.txt", "wb")
    pickle.dump(tabulate.tabulate(table, headers), f)
    f.close()


def main():
    env, Q, hit_table, Q_x = create_environment()
    # Q, hit_table, Q_x = train(total_episodes, max_steps, env, Q, epsilon, hit_table, Q_x)
    # display_table(hit_table)
    # print(Q)
    play()


main()
