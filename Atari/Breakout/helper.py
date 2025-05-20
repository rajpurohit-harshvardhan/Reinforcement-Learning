import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_observation_frame(observation):
    gray_obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    _, threshold_obs = cv2.threshold(gray_obs, 50, 255, cv2.THRESH_BINARY)
    return threshold_obs


def extract_paddle_and_ball(threshold_observation, observation):
    contours, _ = cv2.findContours(threshold_observation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # debug_contours(obs, contours)
    paddle_position = (0, 189)  # Default was None
    ball_position = (0, 0)  # Default was None
    paddle_detected = False

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 188 <= y <= 190 and 10 < w <= 60 and 1 < h < 15:  # Paddle
            paddle_position = (x + w // 2, y)
            paddle_detected = True
        elif 1.5 < w < 3 and 1 < h < 5 and 20 < y < 189:  # Ball
            ball_position = (x + w // 2, y + h // 2)

    if not paddle_detected:
        # print("Paddle not detected in contours. Checking edge cases...")
        # Check for paddle at the far left or far right
        left_edge = threshold_observation[189, 8:10].sum() > 0  # Pixels on the left edge
        right_edge = threshold_observation[189, 145:152].sum() > 0  # Pixels on the right edge

        if left_edge:
            paddle_position = (5, 189)  # Paddle at the leftmost edge
            # print("Paddle detected at the left edge.")

        elif right_edge:
            paddle_position = (145, 189)  # Paddle at the rightmost edge
            # print("Paddle detected at the right edge.")
    return paddle_position, ball_position


def debug_contours(threshold_observation, contours):
    # Convert to color image for easier contours detection
    # vis_image = cv2.cvtColor(threshold_observation, cv2.COLOR_GRAY2BGR)  # Changes color to grayscale
    vis_image = threshold_observation.copy()

    print("Detected Contours:")
    for i, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}")

        color = (0, 25, 0)
        label = ""
        # Mark contours differently based on size (paddle, ball, or unknown)
        if y == 189 and 2 < w < 60 and 1 < h < 15:  # Paddle
            # label = f"Paddle-{i}"
            color = (0, 255, 0)  # Green

        # Ball: Small and above the paddle
        elif 1.5 < w < 3 and 1 < h < 5 and 20 < y < 189:
            color = (255, 0, 0)  # Red
            print(f"Size of Ball :: {w}:{h} ")

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

    plt.imshow(vis_image)
    plt.title("Paddle and Ball Detection")
    plt.show()


def preprocess_frame_and_extract_information(observation):
    threshold_obs = preprocess_observation_frame(observation)
    paddle_position, ball_position = extract_paddle_and_ball(threshold_obs, observation)
    return tuple([paddle_position, ball_position])
