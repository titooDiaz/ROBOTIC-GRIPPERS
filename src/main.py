
import time
import cv2
import joblib
import numpy as np
import serial
import torch

import settings
from src.classifier import GestureClassifier
from src.hand_detector import HAND_CONNECTIONS, HandDetector
from src.robot_movement import RobotMovement


HIGHLIGHT_NODES = set(range(1, settings.THUMB_TIP + 1)) | set(
    range(5, settings.INDEX_TIP + 1)
)


def load_models():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        settings.MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model = GestureClassifier(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint, model, joblib.load(settings.SCALER_PATH), device


def draw_landmarks(frame, results):
    height, width = frame.shape[:2]
    coordinates = None

    for landmarks in results.hand_landmarks:
        points = [
            (int(point.x * width), int(point.y * height))
            for point in landmarks
        ]

        coordinates = points
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], settings.LIGHT, 2)

        for index, point in enumerate(points):
            color = (
                settings.HIGHLIGHT
                if index in HIGHLIGHT_NODES or index == settings.PALM_CENTER
                else settings.LIGHT
            )

            cv2.circle(frame, point, 5, color, -1)

    return coordinates


def predict(hands, checkpoint, model, scaler, device):
    if not hands:
        return "No hand detected", ""

    points = np.asarray(
        hands[0],
        dtype=np.float32
    ).reshape(1, -1)

    if points.shape[1] != checkpoint["input_size"]:
        return "Invalid landmark data", ""

    features = torch.tensor(
        scaler.transform(points),
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        probabilities = torch.softmax(
            model(features),
            dim=1
        )

        confidence, predicted_index = probabilities.max(dim=1)

    confidence = confidence.item()
    confidence_text = f"Confidence: {confidence:.1%}"

    if confidence < settings.CONFIDENCE_THRESHOLD:
        return "Unknown gesture", confidence_text

    return (
        checkpoint["class_names"][predicted_index.item()],
        confidence_text
    )


def draw_text(frame, gesture_text, confidence_text, coordinates, gripper_angle, vertical_angle):
    cv2.rectangle(
        frame,
        (10, 10),
        (530, 190),
        (30, 30, 30),
        -1
    )

    texts = [
        (gesture_text, (25, 40), 0.8, (0, 255, 0), 2),
        (confidence_text, (25, 70), 0.6, settings.LIGHT, 1),
        (f"Gripper: {gripper_angle:.1f} degrees", (25, 105), 0.6, settings.HIGHLIGHT, 2),
        (f"Vertical: {vertical_angle:.1f} degrees", (25, 140), 0.6, settings.HIGHLIGHT, 2),
    ]

    if coordinates:

        thumb_x, thumb_y = coordinates[settings.THUMB_TIP]
        index_x, index_y = coordinates[settings.INDEX_TIP]

        distance = np.linalg.norm(
            np.array([thumb_x, thumb_y])
            - np.array([index_x, index_y])
        )

        palm_x, palm_y = coordinates[settings.PALM_CENTER]

        texts += [
            (f"Finger distance: {distance:.1f} px", (25, 170), 0.55, settings.LIGHT, 1),
        ]

        cv2.circle(
            frame,
            (palm_x, palm_y),
            10,
            (255, 0, 255),
            2
        )

    for text, position, scale, color, thickness in texts:
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness
        )

    cv2.putText(frame, "Press Q to quit", (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, settings.LIGHT, 1)


def main():
    checkpoint, model, scaler, device = load_models()

    print("Loaded gesture classes:", checkpoint["class_names"])
    detector = HandDetector()
    camera = cv2.VideoCapture(settings.CAMERA_INDEX)

    if not camera.isOpened():
        detector.close()
        raise RuntimeError("Could not open the camera.")

    robot = None

    try:
        try:
            robot = RobotMovement()
        except (serial.SerialException, OSError) as error:
            print("ESP32 not found:", error)

        previous_time = time.perf_counter()

        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read camera frame.")

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            # Detect hand
            hands, results = detector.detect(frame)
            # Draw landmarks and movement limits
            coordinates = draw_landmarks(frame, results)
            # Classify gesture
            gesture, confidence = predict(
                hands,
                checkpoint,
                model,
                scaler,
                device
            )

            # Calculate elapsed time
            current_time = time.perf_counter()
            dt = current_time - previous_time
            previous_time = current_time
            distance = None
            hand_y = None

            if coordinates is not None:

                # Finger distance for gripper
                thumb = np.array(coordinates[settings.THUMB_TIP], dtype=np.float32)
                index = np.array(coordinates[settings.INDEX_TIP], dtype=np.float32)
                distance = float(np.linalg.norm(thumb - index))

                # Palm center for vertical movement
                hand_y = coordinates[settings.PALM_CENTER][1]

            # Update both servos
            if robot is not None:
                gripper_angle, vertical_angle = robot.update( distance, hand_y, height, dt)
            else:
                gripper_angle = 0.0
                vertical_angle = 0.0

            # Display information
            draw_text(frame, gesture, confidence, coordinates, gripper_angle, vertical_angle)

            cv2.imshow(
                "Hand Gesture Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

        if robot is not None:
            robot.close()


if __name__ == "__main__":
    main()