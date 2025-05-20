import gymnasium as gym
import cv2
import numpy as np
import ale_py
import matplotlib.pyplot as plt


def preprocess_observation(obs):
    gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    _, threshold_obs = cv2.threshold(gray_obs, 50, 255, cv2.THRESH_BINARY)
    return threshold_obs


def extract_paddle_and_ball(threshold_obs, obs):
    contours, _ = cv2.findContours(threshold_obs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_contours(obs, contours)
    paddle_position = (0, 189)  # Default was None
    ball_position = (0, 0)  # Default was None
    paddle_detected = False

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # if y == 189 and w < 17 and 1 < h < 15:  # Paddle
        if 188 <= y <= 190 and 10 < w <= 60 and 1 < h < 15:
            paddle_position = (x + w // 2, y)
            paddle_detected = True
            print(x,y,w,h)
        elif 1.5 < w < 3 and 1 < h < 5 and 20 < y < 189:
            ball_position = (x + w // 2, y + h // 2)
            # print(f"Size of Ball :: {w}:{h} ")

    if not paddle_detected:
        print("Paddle not detected in contours. Checking edge cases...")
        # Check for paddle at the far left or far right
        left_edge = threshold_obs[189, 8:10].sum() > 0  # Pixels on the left edge
        right_edge = threshold_obs[189, 145:152].sum() > 0  # Pixels on the right edge

        if left_edge:
            paddle_position = (5, 189)  # Paddle at the leftmost edge
            print("Paddle detected at the left edge.")

        elif right_edge:
            paddle_position = (145, 189)  # Paddle at the rightmost edge
            print("Paddle detected at the right edge.")
    return paddle_position, ball_position


def debug_contours(threshold_obs, contours):
    # Convert to color image for visualization
    # vis_image = cv2.cvtColor(threshold_obs, cv2.COLOR_GRAY2BGR)
    vis_image = threshold_obs.copy()
    print(len(contours))

    # print("Detected Contours:")
    for i, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}")

        color = (0, 25, 0)
        label = ""
        # Mark contours differently based on size (paddle, ball, or unknown)
        # if 20 < w < 60 and 5 < h < 15 and y > 180:  # Likely Paddle
        if y == 189 and 2 < w < 60 and 1 < h < 15:  # Paddle
            # label = f"Paddle-{i}"
            color = (0, 255, 0)  # Green
        # Ball: Small and above the paddle
        elif 1.5 < w < 3 and 1 < h < 5 and 20 < y < 189:
            color = (255, 0, 0)  # Red
            # print(f"Size of Ball :: {w}:{h} ")
        else:  # Unknown object
            # label = f"Other-{i}"
            color = (0, 0, 255)  # Blue

        # Draw the bounding box
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

        # Put the label above the contour
        # cv2.putText(vis_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the image with annotations
    plt.imshow(vis_image)
    plt.title("Detected Contours with Labels")
    plt.show()


def visualize_detection(obs, paddle_position, ball_position):
    vis_image = obs.copy()
    if paddle_position:
        cv2.rectangle(vis_image, (paddle_position[0] - 15, paddle_position[1] - 5),
                      (paddle_position[0] + 15, paddle_position[1] + 5), (0, 255, 0), 2)
    if ball_position:
        cv2.circle(vis_image, ball_position, 5, (255, 0, 0), -1)

    cv2.rectangle(vis_image, (0, 0), (8, 196), (0, 255, 0), 1)
    cv2.rectangle(vis_image, (152, 0), (162, 196), (0, 255, 0), 1)
    plt.imshow(vis_image)
    plt.title("Paddle and Ball Detection")
    plt.show()


def choose_actions(ball_pos, paddle_pos):
    # if ball_pos[0] == ball_pos[1] == 0:
    #     action = 1  # Fire action
    # elif (ball_pos[0] < paddle_pos[0] and 35 < paddle_pos[0]) or paddle_pos[0] == 0:
    #     action = 3  # Move paddle left
    # elif ball_pos[0] > paddle_pos[0] and paddle_pos[0] < 140:
    #     action = 2  # Move paddle right
    # else:
    #     action = 0  # Stay
    if ball_pos[0] == ball_pos[1] == 0:
        action = 1  # Fire action
    elif ball_pos[1] < 130:
        action = 0
    elif ball_pos[0] < paddle_pos[0] or paddle_pos[0] == 0:
        action = 3  # Move paddle left
    elif ball_pos[0] > paddle_pos[0] and paddle_pos[0] < 130:
        action = 2  # Move paddle right
    else:
        action = 0  # Stay

    return action


# Main loop
env = gym.make("ALE/Breakout-v5", render_mode="human")
obs, info = env.reset()
print(f"Action space: {env.action_space.n}")
done = False
action = 0  # Random action
for i in range(500):

    obs, reward, terminated, truncated, info = env.step(action)
    # done = terminated or truncated

    # Detect paddle and ball
    threshold_obs = preprocess_observation(obs)
    paddle_position, ball_position = extract_paddle_and_ball(threshold_obs, obs)
    print(f"Paddle: {paddle_position} = {paddle_position[0]}, Ball: {ball_position}, {type(ball_position)}")

    action = choose_actions(ball_position, paddle_position)
    # Visualize
    visualize_detection(obs, paddle_position, ball_position)
    print("shown")

env.close()
