import math
import pickle
import ale_py
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# helper functions import
import helper

alpha = 0.1  # Learning Rate
gamma = 0.9  # Discount RateA

total_episodes = 1000
max_steps = 1000

epsilon = 1
decay_rate = 1 / total_episodes

paddle_position = np.linspace(0, 160, 20)
ball_position_x = np.linspace(0, 160, 15)
ball_position_y = np.linspace(0, 170, 7)
ball_velocity_x = np.linspace(-5, 5, 10)
ball_velocity_y = np.linspace(-5, 5, 20)
ball_predicted_x = np.linspace(0, 160, 20)
# previous_ball_position = (0, 0)  # Initial default
previous_ball_positions = deque(maxlen=5)

print("Observation space ::", tuple([paddle_position, ball_position_x]))


def create_environment(render):
    mode = None
    if render:
        mode = 'human'
    env = gym.make("ALE/Breakout-v5", render_mode=mode)
    Q = np.zeros((len(paddle_position)+1, len(ball_position_x)+1, len(ball_position_y)+1, len(ball_predicted_x)+1, len(ball_velocity_y)+1, env.action_space.n))
    return env, Q


def fetch_ball_and_paddle_info(state, frame_gap=1):
    global previous_ball_positions

    frame_data = helper.preprocess_frame_and_extract_information(state)
    paddle_pos = frame_data[0]
    ball_pos = frame_data[1]

    previous_ball_positions.append(ball_pos)

    if len(previous_ball_positions) > frame_gap:
        reference_ball_pos = previous_ball_positions[-(frame_gap + 1)]  # Get position from 'frame_gap' steps ago
    else:
        reference_ball_pos = previous_ball_positions[0]  # Use oldest available position
        frame_gap = len(previous_ball_positions)

    velocities, predicted_ball_pos_x = calculate_predicted_values(ball_pos, reference_ball_pos, frame_gap)

    return ball_pos, paddle_pos, velocities, predicted_ball_pos_x


def calculate_predicted_values(ball_pos, reference_ball_pos, frame_gap):
    x_min = 8
    x_max = 145

    # Compute velocity based on frame_gap
    velocity_x = (ball_pos[0] - reference_ball_pos[0]) / frame_gap
    velocity_y = (ball_pos[1] - reference_ball_pos[1]) / frame_gap

    predicted_ball_pos_x = 0

    if velocity_y <= 0:
        predicted_ball_pos_x = ball_pos[0]
    else:
        x_intercept = ball_pos[0] + velocity_x * ((187 - ball_pos[1]) / velocity_y)

        while x_intercept < x_min or x_intercept > x_max:  # Handle multiple bounces
            if x_intercept < x_min:
                x_intercept = 2 * x_min - x_intercept
            elif x_intercept > x_max:
                x_intercept = 2 * x_max - x_intercept

        predicted_ball_pos_x = x_intercept
        # if x_intercept < x_min:
        #     predicted_ball_pos_x = x_min + (x_min - x_intercept)
        # elif x_intercept > x_max:
        #     predicted_ball_pos_x = 2 * x_max - x_intercept
        # else:
        #     predicted_ball_pos_x = x_intercept

    return (velocity_x, velocity_y), predicted_ball_pos_x


def choose_action(state, Q, env, ball_pos, paddle_pos, ball_velocity, x_predicted):
    # if np.random.random() < epsilon:
    if ball_pos[0] == ball_pos[1] == 0:  # Fire the ball when life is lost
        return 1  # Fire action

    if ball_velocity[1] <= 0:  # Ball is moving upwards, don't move the paddle
        return 0  # Stay

    # Do not move paddle if ball is very near to paddle's movement axis
    if ball_pos[1] >= 150 and paddle_pos[0] - 10 < x_predicted < paddle_pos[0] + 10:
        return 0

        # Only move the paddle when the ball is coming down
    if (ball_velocity[1] > 0 and ball_pos[1] >= 100) or (ball_velocity[1] >= 30):  # Ball is close to the paddle
        if paddle_pos[0] - 10 < x_predicted < paddle_pos[0] + 10:
            return 0  # Stay
        elif x_predicted > paddle_pos[0] + 8:
            return 2  # Move Right
        elif x_predicted < paddle_pos[0] - 8:
            return 3  # Move Left
        else:
            return 0  # Stay

    return 0  # Default: Stay


def update(state, state2, reward, action, action2, Q):
    predict = Q[state][action]
    target = reward + gamma * Q[state2][action2]
    Q[state][action] = Q[state][action] + alpha * (target - predict)
    return Q


def train(total_episodes, max_steps, env, Q, epsilon):
    data = {}
    max_t = 0
    avg_t = 0
    for episode in range(total_episodes):
        terminated = False
        data[episode] = 0
        t = 0
        lives = 5
        state, info = env.reset()
        ball_pos, paddle_pos, ball_velocity, x_predicted = fetch_ball_and_paddle_info(state, frame_gap=5)
        state_paddle = np.digitize(paddle_pos[0], paddle_position, True)
        state_ball_x = np.digitize(ball_pos[0], ball_position_x, True)
        state_ball_y = np.digitize(ball_pos[1], ball_position_y, True)
        state_ball_velocity_y = np.digitize(ball_velocity[1], ball_velocity_y, True)
        state_predicted_x = np.digitize(x_predicted, ball_predicted_x, True)

        state1 = (state_paddle, state_ball_x, state_ball_y, state_predicted_x, state_ball_velocity_y)

        action1 = choose_action(state1, Q, env, ball_pos, paddle_pos, ball_velocity, x_predicted)

        if episode % (total_episodes / 10) == 0:
            print("#### Episode:", episode, " :: action :", action1, ", Average Steps::", avg_t/(total_episodes/10), "max_steps:", max_t)
            avg_t = 0

        while not terminated:
            # Getting the next state
            state2, reward, done, trunc, info = env.step(action1)
            ball_pos, paddle_pos, ball_velocity, x_predicted = fetch_ball_and_paddle_info(state2, frame_gap=5)
            newstate_paddle = np.digitize(paddle_pos[0], paddle_position, True)
            newstate_ball_x = np.digitize(ball_pos[0], ball_position_x, True)
            newstate_ball_y = np.digitize(ball_pos[1], ball_position_y, True)
            newstate_predicted_x = np.digitize(x_predicted, ball_predicted_x, True)
            newstate_ball_velocity_y = np.digitize(ball_velocity[1], ball_velocity_y, True)

            state2 = (newstate_paddle, newstate_ball_x, newstate_ball_y, newstate_predicted_x, newstate_ball_velocity_y)

            # Choosing the next action
            action2 = choose_action(state2, Q, env, ball_pos, paddle_pos, ball_velocity, x_predicted)

            # Learning the Q-value
            Q = update(state1, state2, reward, action1, action2, Q)

            state1 = state2
            action1 = action2

            # Updating the respective values
            t += 1

            if reward > 0:
                reward += 5000  # Encourage hitting the bricks
            elif info["lives"] < lives:
                reward -= 5000  # Penalize losing a life
                lives -= 1
            elif reward == 0:
                reward = -10

            # If at the end of learning process
            if done:
                terminated = True
                data[episode] = reward
                if t > max_t:
                    max_t = t
                avg_t += t
                break

        epsilon = max(epsilon - decay_rate, 0)

    # saving the Q table in a file
    f = open("breakout.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    plot_graph(data)

    return Q


def play():
    env = gym.make("ALE/Breakout-v5")

    f = open("breakout_10k_new.pkl", "rb")  # Read the Q table from the File
    Q = pickle.load(f)
    f.close()

    loop_for = 10
    rewards_avg = 0

    for i in range(loop_for):
        episode_terminated = False
        lives = 5
        t = 0

        reset_state, info = env.reset()
        ball_pos, paddle_pos, ball_velocity, x_predicted = fetch_ball_and_paddle_info(reset_state, frame_gap=5)
        state_paddle = np.digitize(paddle_pos[0], paddle_position, True)
        state_ball_x = np.digitize(ball_pos[0], ball_position_x, True)
        state_ball_y = np.digitize(ball_pos[1], ball_position_y, True)
        state_predicted_x = np.digitize(x_predicted, ball_predicted_x, True)
        state_ball_velocity_y = np.digitize(ball_velocity[1], ball_velocity_y, True)
        state = (state_paddle, state_ball_x, state_ball_y, state_predicted_x, state_ball_velocity_y)

        rewards = 0
        action = np.argmax(Q[state])

        while not episode_terminated:
            # print("X_predicted ::", x_predicted)

            state2, reward, done, trunc, info = env.step(action)
            ball_pos, paddle_pos, ball_velocity, x_predicted = fetch_ball_and_paddle_info(state2, frame_gap=5)
            newstate_paddle = np.digitize(paddle_pos[0], paddle_position, True)
            newstate_ball_x = np.digitize(ball_pos[0], ball_position_x, True)
            newstate_ball_y = np.digitize(ball_pos[1], ball_position_y, True)
            newstate_predicted_x = np.digitize(x_predicted, ball_predicted_x, True)
            newstate_ball_velocity_y = np.digitize(ball_velocity[1], ball_velocity_y, True)
            state = (newstate_paddle, newstate_ball_x, newstate_ball_y, newstate_predicted_x, newstate_ball_velocity_y)

            t += 1
            rewards += reward

            action = np.argmax(Q[state])

            if lives != info["lives"] or ball_pos[0] == ball_pos[1] == 0:
                action = 1

            if lives != info["lives"]:
                lives -= 1

            if done:
                print("Rewards ::", rewards)
                rewards_avg += rewards
                episode_terminated = True
                break

    print(f"Average rewards after {loop_for} loops : {rewards_avg / loop_for}")


def plot_graph(data):
    lists = sorted(data.items())  # sorted by key, return a list of tuples
    lists2 = sorted(data.values())  # sorted by values, return a list of values
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    # plt.show()
    # plt.plot(lists2)
    # plt.show()


def main():
    env,Q = create_environment(False)
    # print(env, Q)
    # Q = train(total_episodes, max_steps, env, Q, epsilon)
    play()


main()