import gymnasium as gym
import numpy as np
import pickle
import tabulate
import math
import matplotlib.pyplot as plt

alpha = 0.1  # Learning Rate
gamma = 0.9  # Discount Rate

total_episodes = 100000
max_steps = 1000
min_epsilon = 0.001
max_epsilon = 1.0
decay_rate = 1/total_episodes
epsilon = 1

horizontal_position_bins = np.linspace(-0.5, 0.5, 7)  # [far_left, left, center, right, far_right]
vertical_position_bins = np.linspace(-0.2, 1.5, 7)   # [low, medium, high]
horizontal_speed_bins = np.linspace(-1, 1, 7)     # [fast_left, slow_left, stopped, slow_right, fast_right]
vertical_speed_bins = np.linspace(-1, 0.6, 15)       # [fast_descent, slow_descent, hovering, ascending]
angle_bins = np.linspace(-0.4, 0.4, 7)            # [tilted_left, level, tilted_right]
angular_speed_bins = np.linspace(-1, 1, 14)        # [spinning_left, stable, spinning_right]
leg_contact_bins = [0, 1, 2]                      # [no_contact, one_leg_contact, both_legs_contact]


# This function initializes the Gymnasium environment and creates a structure for Q table
def create_environment():
    env = gym.make('LunarLander-v2')

    # here Length+1 is done because in certain episodes the state values goes beyond the defined endpoints.
    Q = np.zeros((len(vertical_speed_bins), len(angular_speed_bins),  env.action_space.n))
    return env, Q


# This function helps with selection of the action based on the state and epsilon value
def choose_actions(state, env):
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
    # if 0:
        action = env.action_space.sample()
    else:

        angle_targ = x_distance * 0.5 + x_velocity * 1.0  # angle should point towards center
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        hover_targ = 0.55 * np.abs(x_distance)  # target y should be proportional to horizontal offset

        angle_todo = (angle_targ - angle) * 0.5 - angular_velocity * 1.0
        hover_todo = (hover_targ - y_distance) * 0.5 - y_velocity * 0.5

        if state[6] or state[7]:  # legs have contact
            angle_todo = 0
            hover_todo = (-y_velocity * 0.5)  # override to reduce fall speed, that's all we need after contact

        if env.continuous:
            a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
            a = np.clip(a, -1, +1)
        else:
            a = 0
            if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
                a = 2
            elif angle_todo < -0.05:
                a = 3
            elif angle_todo > +0.05:
                a = 1
        return a
        # else:

    return action

# This function is responsible for updating the Q-value in the Q table for the given State-Action pair
def update(state1, state2, reward, action, action2, Q):
    predict = Q[state1][action]
    target = reward + gamma * Q[state2][action2]
    a = Q[state1][action]
    b = alpha * (target - predict)
    Q[state1][action] = a + b
    return Q


# This function is used for binning values in the defined bins/categories in the start using np.linspace
def discretize_state(state):
    horizontal_position = int(np.digitize(state[0], horizontal_position_bins) - 1)
    vertical_position = int(np.digitize(state[1], vertical_position_bins) - 1)
    horizontal_speed = int(np.digitize(state[2], horizontal_speed_bins) - 1)
    vertical_speed = int(np.digitize(state[3], vertical_speed_bins) - 1)
    angle = int(np.digitize(state[4], angle_bins) - 1)
    angular_speed = int(np.digitize(state[5], angular_speed_bins) - 1)

    return vertical_speed, angular_speed



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
    # else: total_reward += x_distance
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
    # else: total_reward += 1
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
    #     else: total_reward -= 100

    # High altitude adjustments
    if y_distance > 0.5:
        total_reward += -abs(x_velocity) * 10  # Light penalty for x_velocity

    # Mid altitude adjustments
    elif 0.1 < y_distance <= 0.5:
        total_reward += -abs(x_velocity) * 20  # Stronger penalty for horizontal drift
        total_reward += -abs(angle) * 20  # Penalty for any tilt

    # Low altitude adjustments
    else:
        total_reward += -abs(
            y_velocity) * 20 if y_velocity < -0.1 else y_velocity * 10  # High penalty for fast descent, reward slow descent
        total_reward += -abs(x_velocity) * 10 if abs(x_distance) > 0.05 else -abs(
            x_velocity) * 200  # Strong incentive to stay near center

    # Stabilize angle and angular velocity
    total_reward += -abs(angle) * 10
    total_reward += -abs(angular_velocity) * 5

    # Reward for stable near-landing conditions
    if -0.1 <= x_distance <= 0.1 and -0.05 <= angle <= 0.05 and -0.1 < y_velocity < 0.1:
        total_reward += 500  # Strong incentive for stable, center-aligned descent

    # Large reward for perfect landing
    if -0.05 < x_distance < 0.05 and -0.1 < y_velocity < 0.1 and -0.05 < angle < 0.05:
        total_reward += 10000
        # elif state2[6] == 1 or state1[6] == 1:
    #     if y_velocity < -0.1:
    #         total_reward += -100
    #     else:
    #         total_reward += -100

    return total_reward

def reward_action_formula(reward, state2):
    total_reward = reward

    x_distance = state2[0]
    y_distance = state2[1]
    x_velocity = state2[2]
    y_velocity = state2[3]
    angle = state2[4]
    angular_velocity = state2[5]

    if angle > 0.4 or angle < -0.4:
        total_reward -= 10

    total_reward += -100 * math.sqrt(math.pow(x_distance, 2) + math.pow(y_distance, 2))
    total_reward += -100 * math.sqrt(math.pow(x_velocity, 2) + math.pow(y_velocity, 2))
    total_reward += -100 * math.fabs(angular_velocity)
    total_reward += 10 * state2[6]
    total_reward += 10 * state2[7]

    if -0.01 < x_distance < 0.01:
        total_reward += 100
    else:
        total_reward -= 10

    if y_distance < 0.01:
        total_reward += 100
    else:
        total_reward -= 10

    if state2[6] == 1 and state2[7] == 1:
        if -0.4 < angle < 0.4 and -0.01 < y_velocity < 0.01:
            # print("SUccessfull Landing", round(y_velocity, 2), angle,  round(x_distance, 2))
            total_reward += 100

            if y_distance == 0:
                print("SUccessfull Landing", round(y_velocity, 2), angle,  round(x_distance, 2))
                total_reward += 1000
            else:
                total_reward -= 100
        else:
            total_reward -= 100

    # elif state2[6] == 1 or state1[6] == 1:
    #     if y_velocity < -0.1:
    #         total_reward += -100
    #     else:
    #         total_reward += -100

    return total_reward

# This function is responsible for the training of the agent
def train(total_episodes, max_steps, env, Q, epsilon):
    data = {}
    for episode in range(total_episodes):
        data[episode] = 0
        rewards = 0
        t = 0
        state1 = env.reset()[0]  # resetting the environment
        digitized_state_1 = discretize_state(state1)  # binning state value

        action1 = choose_actions(state1, env)  # choosing an action based on the state

        if episode % (total_episodes/10) == 0:
            print("#### Episode:", episode, " :: action :", action1, ", Epsilon Value::", epsilon)


        episode_finished = False
        while not episode_finished:
            state2, reward, done, trunc, info = env.step(action1)  # Taking a step using environment

            digitized_state_2 = discretize_state(state2)  # binning state value
            action2 = choose_actions(state2, env)   # choosing an action based on the state


            # if digitized_state_2 == 0:
            #     print("Episode :: ", episode, ", state_value :", state2[3])

            # reward = reward_action(reward, state2)
            reward = reward_action_formula(reward, state2)

            Q = update(digitized_state_1, digitized_state_2, reward, action1, action2, Q)  # updating the Q Value

            action1 = action2  # using the action deduced earlier for taking the next step.
            t+=1
            rewards += reward
            if done or trunc:
                data[episode] = rewards
                episode_finished = True
                # print("Episode finished with reward value ::: ", rewards)
                rewards = 0
                break

            if t>max_steps:
                print("Exceeded max steps limit")
                episode_finished = True
                break

        # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        epsilon = max(epsilon - decay_rate, 0)

    # saving the Q table in a file
    f = open("lunar_lander_heuristic.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    lists = sorted(data.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    # plt.show()

    return Q


def calculate_angle(state, action):
    x_distance = state[0]
    y_distance = state[1]
    x_velocity = state[2]
    y_velocity = state[3]
    angle = state[4]
    angular_velocity = state[5]

    angle_targ = x_distance * 0.5 + x_velocity * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(x_distance)  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - angle) * 0.5 - angular_velocity * 1.0
    hover_todo = (hover_targ - y_distance) * 0.5 - y_velocity * 0.5

    if state[6] or state[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (-y_velocity * 0.5)  # override to reduce fall speed, that's all we need after contact

    a = action
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif angle_todo < -0.05:
        a = action
    elif angle_todo > +0.05:
        a = 1
    return a

# this function is responsible for rendering the agent playing the game using the Q table we computed in the training
def play(Q):
    env = gym.make('LunarLander-v2', render_mode='human')

    # Read the Q table from the File
    f = open("lunar_lander_heuristic.pkl", "rb")
    Q = pickle.load(f)
    f.close()

    for i in range(20):
        print('#'*50)
        t = 0
        total_rewards = 0
        state1 = env.reset()[0]
        digitized_state_1 = discretize_state(state1)  # binning state value
        # print("STARTING STATE :::: ", state1, digitized_state_1)

        action1 = np.argmax(Q[digitized_state_1])  # since Q table stores the values with digitized state

        episode_finished = False
        counter = 0
        while not episode_finished:
            state2, reward, done, trunc, info = env.step(action1)  # Taking a step using environment

            digitized_state_2 = discretize_state(state2)  # binning state value

            action2 = np.argmax(Q[digitized_state_2])  # since Q table stores the values with digitized state
            # action2 = calculate_angle(state2, action2)
            # if action2 == 0:
            #     print("doing nothing.", Q[digitized_state_2], digitized_state_2, state2)

            action1 = action2  # using the action deduced earlier for taking the next step.
            t += 1
            total_rewards += reward

            if done:
                episode_finished = True
                print("Episode finished with reward value ::: ", total_rewards)
                break

            if t>max_steps:
                print("Exceeded max steps limit")
                episode_finished = True
                break


# def display_table(hit_table):
#     #  VALUES IN TOP = lander_angle values
#     #  Values going down = Velocity in Y direction
#
#     table = []
#     velocity_y = np.append(lander_velocity_y, math.inf)
#     angular_velocity = np.append(lander_angle, math.inf)
#
#     for i in range(len(velocity_y)):
#         # record = [math.log(item, 10) if item > 0 else item for item in hit_table[i]]
#         record = [ item for item in hit_table[i]]
#         record = np.append(velocity_y[i], record)
#         table.append(record)
#
#     headers = np.append("Velocity in Y direction", angular_velocity)
#
#     # saving the hit table in a file
#     f = open("lunar_lander_heuristic.txt", "wb")
#     pickle.dump(tabulate.tabulate(table, headers), f)
#     f.close()


def main():
    env, Q = create_environment()
    # Q = train(total_episodes, max_steps, env, Q, epsilon)
    # display_table(hit_table) 
    # print(Q)
    play(Q)


main()
