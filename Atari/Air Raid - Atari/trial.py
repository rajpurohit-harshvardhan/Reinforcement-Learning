import cv2
import pickle
import ale_py
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# gym.register_envs(ale_py)

# Initialize the environment
env = gym.make('ALE/AirRaid-v5')

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor

total_episodes = 1000  # Number of episodes
max_steps = 10000

epsilon = 1
decay_rate = 1 / total_episodes

screen_width = 160  # Width of the screen

player_pos = np.linspace(0, 9, 10)
player_velocity = np.linspace(2, 11, 10)
can_shoot = np.linspace(0, 1, 2)
distance_to_left_edge = np.linspace(0, 100, 10)
distance_to_right_edge = np.linspace(0, 100, 10)

print("Observation space ::", tuple([player_pos, player_velocity, can_shoot, distance_to_left_edge, distance_to_right_edge]))

def create_environment():
    env = gym.make('AirRaid-v4')
    Q = np.zeros((len(player_pos)+1, len(player_velocity)+1, len(can_shoot), len(distance_to_left_edge)+1, len(distance_to_right_edge)+1,  env.action_space.n))
    return env, Q


def preprocess_frame(frame, resize_shape=(84, 84), threshold_value=200):
    """
    Preprocesses a game frame: converts to grayscale, resizes, and applies thresholding.

    Parameters:
        frame (np.ndarray): Original RGB frame from the environment.
        resize_shape (tuple): Target shape for resizing (width, height).
        threshold_value (int): Threshold for binarizing the frame.

    Returns:
        np.ndarray: Preprocessed binary image.
    """
    print(type(frame), frame.shape)
    # Convert to grayscale
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif len(frame.shape) == 2:  # Already grayscale
        gray_frame = frame
    else:
        raise ValueError("Unexpected frame shape: ", frame.shape)
    # Resize the frame
    resized_frame = cv2.resize(gray_frame, resize_shape)
    # Apply thresholding
    _, binary_frame = cv2.threshold(resized_frame, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_frame


def extract_player_info(binary_frame, prev_position, screen_width=84):
    """
    Extracts player-specific information from a preprocessed frame.

    Parameters:
        binary_frame (np.ndarray): Binary image after preprocessing.
        prev_position (tuple or None): Player's position from the previous frame (x, y).
        screen_width (int): Width of the screen for calculating boundary distances.

    Returns:
        dict: Player information including position, velocity, and boundary distances.
    """

    # Find contours
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Identify the largest contour (assume it's the player)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        current_position = (x + w // 2, y + h // 2)  # Center of the bounding box
    else:
        current_position = None

    # Calculate velocity
    if current_position and prev_position:
        velocity = current_position[0] - prev_position[0]  # Horizontal velocity
    else:
        velocity = 0  # No velocity on the first frame or if player is not detected

    # Calculate boundary distances
    if current_position:
        distance_to_left = current_position[0]
        distance_to_right = screen_width - current_position[0]
    else:
        distance_to_left = distance_to_right = 0  # Default if no player detected

    return {
        "position": current_position,
        "velocity": velocity,
        "distance_to_left": distance_to_left,
        "distance_to_right": distance_to_right
    }


def choose_action(state, env, Q):
    """
    Choose an action using the epsilon-greedy policy.
    Args:
        state: Current state index (integer).
        epsilon: Exploration rate (float between 0 and 1).
    Returns:
        Chosen action index (integer).
    """
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Explore: random action
    return np.argmax(Q[state])  # Exploit: action with the highest Q-value


def digitize_states(state, prev_position):
    # TODO: Digitize state spaces
    binary_frame = preprocess_frame(state)

    # Extract player info
    player_info = extract_player_info(binary_frame, prev_position, screen_width)
    prev_position = player_info["position"]  # Update the previous position

    # Create the state vector
    if player_info["position"] is not None:
        digitized_state = [
            np.digitize(player_info["position"][0], player_pos) if player_info["position"] else 0,
            np.digitize(player_info["velocity"], player_velocity),
            np.digitize(player_info["distance_to_left"], distance_to_left_edge),
            np.digitize(player_info["distance_to_right"], distance_to_right_edge)
        ]
    else:
        digitized_state = [0, 0, 0, 0]  # Default state if player not detected
    return tuple(digitized_state), prev_position

def update(state, state2, reward, action, action2, Q):
    predict = Q[state][action]
    target = reward + gamma * Q[state2][action2]
    Q[state][action] = Q[state][action] + alpha * (target - predict)
    return Q


def train(total_episodes, max_steps, env, Q, epsilon):
    data = {}
    # Q-learning loop
    for episode in range(total_episodes):
        data[episode] = 0
        t = 0
        done = False

        prev_position = None
        state = env.reset()
        state1, prev_position = digitize_states(state[0], prev_position)
        action1 = choose_action(state1, env, Q)

        if episode % (total_episodes / 10) == 0:
            print("#### Episode:", episode, " :: action :", action1, ", Epsilon Value::", epsilon)

        while t < max_steps:
            state2, reward, done, _, _ = env.step(action1)
            state2 = digitize_states(state2, prev_position)
            action2 = choose_action(state2, env, Q)

            # Update Q-table (add Q-learning logic here)
            Q = update(state1, state2, reward, action1, action2, Q)

            state1 = state2
            action1 = action2
            t += 1

            if done:
                data[episode] = reward
                break

        epsilon = max(epsilon - decay_rate, 0)

    # saving the Q table in a file
    f = open("air_raid_1.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    # lists = sorted(data.items())  # sorted by key, return a list of tuples
    # lists2 = sorted(data.values())  # sorted by values, return a list of values
    # x, y = zip(*lists)  # unpack a list of pairs into two tuples
    # plt.plot(x, y)
    # plt.show()
    # plt.plot(lists2)
    # plt.show()

    return Q


def play():
    env = gym.make('AirRaid-v4')

    # Read the Q table from the File
    f = open("air_raid_1.pkl", "rb")
    Q = pickle.load(f)
    f.close()

    for i in range(10000):
        t = 0
        prev_position = None
        reset_state = env.reset()[0]
        state, prev_position = digitize_states(reset_state, prev_position)
        rewards = 0

        while t < max_steps:
            state2, reward, done, trunc, info = env.step(np.argmax(Q[state]))
            state, prev_position = digitize_states(reset_state, prev_position)

            t += 1
            rewards += reward

            if done:
                print("Rewards ::", rewards)
                break


def main():
    env, Q = create_environment()
    _ = train(total_episodes, max_steps, env, Q, epsilon)
    play()


main()
