import math
import pickle
import random
import ale_py
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

alpha = 0.1  # Learning Rate
gamma = 0.9  # Discount Rate

total_episodes = 1000
max_steps = 300

epsilon = 1
decay_rate = 1 / total_episodes

paddle_position = np.linspace(0, 160, 20)
ball_position_x = np.linspace(0, 160, 20)
ball_position_y = np.linspace(0, 170, 20)
ball_velocity_x = np.linspace(-5, 5, 10)
ball_velocity_y = np.linspace(-2, 2, 5)
# ball_velocity_y = np.linspace(-1, 20, 20)
ball_predicted_x = np.linspace(0, 160, 20)

previous_ball_positions = deque(maxlen=5)
old_ball_position = (0, 0)

print("Observation space ::", tuple([paddle_position, ball_position_x]))


def create_environment(render):
    mode = None
    if render:
        mode = 'human'
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=mode)
    # Q = np.zeros((len(paddle_position)+1, len(ball_position_x)+1, len(ball_position_y)+1, len(ball_predicted_x)+1, len(ball_velocity_y)+1, env.action_space.n))
    Q = np.zeros((len(paddle_position)+1, len(ball_predicted_x)+1, len(ball_position_y)+1, len(ball_velocity_y)+1, env.action_space.n))
    return env, Q


def fetch_information_from_ram(env):
    ram_state = env.unwrapped.ale.getRAM()
    paddle = ram_state[72]
    ball = (ram_state[99], ram_state[101])
    life = ram_state[57]
    score = ram_state[84]
    return paddle, ball, life, score


calculated_position = 0


def calculate_predicted_position(ball_pos, frame_gap):
    global old_ball_position
    global calculated_position
    if not calculated_position:
        calculated_position = ball_pos[0]

    x_min = 50
    x_max = 190

    velocity_x = (float(ball_pos[0]) - float(old_ball_position[0])) / frame_gap
    velocity_y = (float(ball_pos[1]) - float(old_ball_position[1])) / frame_gap

    predicted_ball_pos_x = 0

    if velocity_y <= 0:
        predicted_ball_pos_x = ball_pos[0]
    else:
        x_intercept = ball_pos[0] + velocity_x * ((182 - ball_pos[1]) / velocity_y)

        while x_intercept < x_min or x_intercept > x_max:  # Handle multiple bounces
            if x_intercept < x_min:
                x_intercept = 2 * x_min - x_intercept
            elif x_intercept > x_max:
                x_intercept = 2 * x_max - x_intercept

        predicted_ball_pos_x = x_intercept

    old_ball_position = ball_pos
    # if ball_pos[1] >= 125 or reference_ball_pos[0] == 0:
    #     predicted_ball_pos_x = calculated_position
    # else:
    #     calculated_position = predicted_ball_pos_x
    #   if x_intercept < x_min:
    #     predicted_ball_pos_x = x_min + (x_min - x_intercept)
    # elif x_intercept > x_max:
    #     predicted_ball_pos_x = 2 * x_max - x_intercept
    # else:
    #     predicted_ball_pos_x = x_intercept

    return (velocity_x, velocity_y), predicted_ball_pos_x


def choose_action(state, Q, env, ball_pos, paddle_pos, ball_velocity, x_predicted, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        if ball_pos[1] == 0:  # Fire the ball when life is lost
            return 1  # Fire action

        if ball_velocity[1] < 0:  # Ball is moving upwards, don't move the paddle
            return 0  # Stay

        # if (ball_velocity[1] > 0 and ball_pos[1] >= 100) or (ball_velocity[1] >= 30):  # Ball is close to the paddle
        if ball_velocity[1] > 0:  # Ball is close to the paddle
            if ball_pos[1] > 180:
                if paddle_pos < ball_pos[0] < paddle_pos + 14:
                    return 0  # Stay
                elif ball_pos[0] > paddle_pos + 14:
                    return 2  # Move Right
                elif ball_pos[0] < paddle_pos:
                    return 3  # Move Left
                else:
                    return 0  # Stay

            if paddle_pos == 55 and 50 < x_predicted < 65:
                return 0
            # if 182 < paddle_pos < 190 and 182 < x_predicted < 199:
            #     return 0

            if paddle_pos < x_predicted < paddle_pos + 14:
                return 0  # Stay
            elif x_predicted > paddle_pos + 14:
                return 2  # Move Right
            elif x_predicted < paddle_pos:
                return 3  # Move Left
            else:
                return 0  # Stay

        return 0  # Default: Stay


def update(state, state2, reward, action, action2, Q):
    predict = Q[state][action]
    target = reward + gamma * Q[state2][action2]
    Q[state][action] = Q[state][action] + alpha * (target - predict)
    return Q


def reward_agent(reward, paddle, ball, x_predicted, ball_velocity, action, life_lost):
    if life_lost:  # Penalizing agent for losing a life
        reward = -100
    else:
        reward = 0
        # if reward > 0:  # Meaning agent hit a brick
        #     reward = 100
        # else:
        #     reward = 0

    if ball_velocity[1] > 0:  # Rewarding only when ball is moving down
        # if paddle == 55 and 50 < x_predicted < 65 and action == 0:
        #     reward += 0.5

        if paddle < x_predicted < paddle + 14 and action == 0:
            reward += 50  # Stay
        elif x_predicted > paddle + 14 and action == 2:
            # reward += x_predicted - (paddle + 14)  # Move Right
            reward += 0.5  # Move Right
        elif x_predicted < paddle and action == 3:
            # reward += paddle - x_predicted  # Move Left
            reward += 0.5  # Move Left
        # elif action == 0:
        #     reward += 0.1  # Stay
        else:
            reward -= 1
    else:
        if action != 0:
            reward -= 1
        else:
            reward += 0.01
    return reward


def train(total_episodes, max_steps, env, Q, epsilon):
    global old_ball_position

    data = {}
    max_t = 0
    avg_t = 0
    max_reward = 0
    previous_score = 0

    for episode in range(total_episodes):
        steps_to_take_actions_to_avoid_deadloop = 0
        terminated = False
        data[episode] = 0
        t = 0
        steps_to_detect_deadloop = 0
        lives = 5
        life_lost = False
        total_score = 0
        accumulated_reward = 0
        rewards = 0

        env.reset()
        paddle, ball, life, score = fetch_information_from_ram(env)
        ball_velocity, x_predicted = calculate_predicted_position(ball, frame_gap=4)
        state_paddle = np.digitize(paddle, paddle_position, True)
        state_ball_x = np.digitize(ball[0], ball_position_x, True)
        state_ball_y = np.digitize(ball[1], ball_position_y, True)
        state_ball_velocity_y = np.digitize(ball_velocity[1], ball_velocity_y, True)
        state_predicted_x = np.digitize(x_predicted, ball_predicted_x, True)

        # state1 = (state_paddle, state_ball_x, state_ball_y, state_predicted_x, state_ball_velocity_y)
        state1 = (state_paddle, state_predicted_x, state_ball_y, state_ball_velocity_y)

        action1 = choose_action(state1, Q, env, ball, paddle, ball_velocity, x_predicted, epsilon)

        if episode % (total_episodes/10) == 0:
            print("#### Episode:", episode, ", Average Steps::", avg_t/(total_episodes/10), "max_steps:", max_t)
            avg_t = 0

        while not terminated:
            # Getting the next state
            state2, reward, done, trunc, info = env.step(action1)

            rewards += reward

            paddle, ball, life, score = fetch_information_from_ram(env)
            ball_velocity, x_predicted = calculate_predicted_position(ball, frame_gap=4)
            newstate_paddle = np.digitize(paddle, paddle_position, True)
            newstate_ball_x = np.digitize(ball[0], ball_position_x, True)
            newstate_ball_y = np.digitize(ball[1], ball_position_y, True)
            newstate_predicted_x = np.digitize(x_predicted, ball_predicted_x, True)
            newstate_ball_velocity_y = np.digitize(ball_velocity[1], ball_velocity_y, True)

            # state2 = (newstate_paddle, newstate_ball_x, newstate_ball_y, newstate_predicted_x, newstate_ball_velocity_y)
            state2 = (newstate_paddle, newstate_predicted_x, newstate_ball_y, newstate_ball_velocity_y)

            # Choosing the next action
            action2 = choose_action(state2, Q, env, ball, paddle, ball_velocity, x_predicted, epsilon)

            if life != lives:
                lives -= 1
                life_lost = True
            else:
                life_lost = False

            reward = reward_agent(reward, paddle, ball, x_predicted, ball_velocity, action2, life_lost)
            accumulated_reward += reward

            if t % 4 == 0:
                Q = update(state1, state2, reward, action1, action2, Q)
                accumulated_reward = 0

            state1 = state2
            action1 = action2

            if score == previous_score:
                steps_to_detect_deadloop += 1
            else:
                previous_score = score
                steps_to_detect_deadloop = 0

            if steps_to_detect_deadloop > max_steps:  # to handle dead loop
                action1 = 0
                steps_to_detect_deadloop = 0
                steps_to_take_actions_to_avoid_deadloop = 30

            if steps_to_take_actions_to_avoid_deadloop > 0:
                if True:
                    action1 = env.action_space.sample()
                else:
                    action1 = np.argmax(Q[state2])
                steps_to_take_actions_to_avoid_deadloop -= 1

            if t > 25000:
                action1 = 0

            # Updating the respective values
            t += 1

            # If at the end of learning process
            if done:
                # print("Steps:", t)
                terminated = True
                data[episode] = rewards
                if t > max_t:
                    max_t = t
                if rewards>max_reward:
                    max_reward = rewards
                avg_t += t
                break

        epsilon = max(epsilon - decay_rate, 0)

    # saving the Q table in a file
    f = open("breakout_v4_ee.pkl", "wb")
    pickle.dump(Q, f)
    f.close()
    print("Max rewards ::", max_reward)
    plot_graph(data)

    return Q


def play(render):
    mode = None
    if render:
        mode = 'human'
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=mode)

    f = open("breakout_v4_ee.pkl", "rb")  # Read the Q table from the File
    Q = pickle.load(f)
    f.close()

    loop_for = 10
    rewards_avg = 0

    for i in range(loop_for):
        episode_terminated = False
        lives = 5
        t = 0

        reset_state, info = env.reset()
        paddle, ball, life, score = fetch_information_from_ram(env)
        ball_velocity, x_predicted = calculate_predicted_position(ball, frame_gap=4)
        state_paddle = np.digitize(paddle, paddle_position, True)
        state_ball_x = np.digitize(ball[0], ball_position_x, True)
        state_ball_y = np.digitize(ball[1], ball_position_y, True)
        state_predicted_x = np.digitize(x_predicted, ball_predicted_x, True)
        state_ball_velocity_y = np.digitize(ball_velocity[1], ball_velocity_y, True)


        state = (state_paddle, state_ball_x, state_ball_y, state_predicted_x, state_ball_velocity_y)
        state = (state_paddle, state_predicted_x, state_ball_y, state_ball_velocity_y)

        rewards = 0
        action = np.argmax(Q[state])

        while not episode_terminated:
            # print("X_predicted ::", x_predicted)

            state2, reward, done, trunc, info = env.step(action)
            paddle, ball, life, score = fetch_information_from_ram(env)
            ball_velocity, x_predicted = calculate_predicted_position(ball, frame_gap=4)
            newstate_paddle = np.digitize(paddle, paddle_position, True)
            newstate_ball_x = np.digitize(ball[0], ball_position_x, True)
            newstate_ball_y = np.digitize(ball[1], ball_position_y, True)
            newstate_predicted_x = np.digitize(x_predicted, ball_predicted_x, True)
            newstate_ball_velocity_y = np.digitize(ball_velocity[1], ball_velocity_y, True)

            state = (newstate_paddle, newstate_ball_x, newstate_ball_y, newstate_predicted_x, newstate_ball_velocity_y)
            state = (newstate_paddle, newstate_predicted_x, newstate_ball_y, newstate_ball_velocity_y)

            t += 1
            if reward > 0:
                rewards += 1

            action = np.argmax(Q[state])
            abc = Q[state]

            if life != lives or ball[0] == ball[1] == 0:
                action = 1

            if life != lives:
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
    plt.show()
    plt.plot(lists2)
    plt.show()


def main():
    env,Q = create_environment(False)
    Q = train(total_episodes, max_steps, env, Q, epsilon)
    play(False)


main()