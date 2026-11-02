from pathlib import Path
import cv2
import joblib
import numpy as np
import torch
from src.classifier import GestureClassifier
from src.hand_detector import HAND_CONNECTIONS, HandDetector


import time
from src.robot_movement import RobotMovement


MODEL_PATH = Path("models/gesture_model.pth")
SCALER_PATH = Path("models/scaler.joblib")
CONFIDENCE_THRESHOLD = 0.60
LIGHT = (220, 220, 220)
HIGHLIGHT = (0, 220, 255)
THUMB_TIP = 4
INDEX_TIP = 8
HIGHLIGHT_NODES = set(range(1, THUMB_TIP + 1)) | set(
    range(5, INDEX_TIP + 1)
)


def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = GestureClassifier(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model, joblib.load(SCALER_PATH), device


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
            cv2.line(frame, points[start], points[end], LIGHT, 2)

        for index, point in enumerate(points):
            color = HIGHLIGHT if index in HIGHLIGHT_NODES else LIGHT
            cv2.circle(frame, point, 5, color, -1)

    return coordinates


def predict(hands, checkpoint, model, scaler, device):
    if not hands:
        return "No hand detected", ""

    points = np.asarray(hands[0], dtype=np.float32).reshape(1, -1)
    if points.shape[1] != checkpoint["input_size"]:
        return "Invalid landmark data", ""

    features = torch.tensor(
        scaler.transform(points),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(features), dim=1)
        confidence, predicted_index = probabilities.max(dim=1)

    confidence = confidence.item()
    confidence_text = f"Confidence: {confidence:.1%}"
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown gesture", confidence_text
    return checkpoint["class_names"][predicted_index.item()], confidence_text


def draw_text(frame, gesture_text, confidence_text, coordinates):
    cv2.rectangle(frame, (10, 10), (520, 120), (30, 30, 30), -1)
    texts = [
        (gesture_text, (25, 48), 0.9, (0, 255, 0), 2),
        (confidence_text, (25, 80), 0.6, (255, 255, 255), 1),
    ]
    if coordinates:
        thumb_x, thumb_y = coordinates[THUMB_TIP]
        index_x, index_y = coordinates[INDEX_TIP]
        distancia = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
        texts += [
            (f"Distancia entre los dedos: {distancia:.2f}", (25, 112), 0.55, HIGHLIGHT, 1),
        ]

    for text, position, scale, color, thickness in texts:
        cv2.putText(
            frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness,
        )

    cv2.putText(
        frame, "Press Q to quit", (15, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    )


def main():
    checkpoint, model, scaler, device = load_models()
    print("Loaded gesture classes:", checkpoint["class_names"])
    detector = HandDetector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        detector.close()
        raise RuntimeError("Could not open the camera.")

    robot = None

    try:
        robot = RobotMovement()
        previous_time = time.perf_counter()

        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a camera frame.")
            frame = cv2.flip(frame, 1)

            # Detect hand
            hands, results = detector.detect(frame)

            # Draw landmarks
            coordinates = draw_landmarks(frame, results)

            # Classify gesture
            gesture, confidence = predict(
                hands, checkpoint, model, scaler, device
            )

            # Calculate elapsed time
            current_time = time.perf_counter()

            dt = current_time - previous_time
            previous_time = current_time

            # Calculate finger distance
            distance = None

            if coordinates is not None:

                thumb = np.array(
                    coordinates[THUMB_TIP],
                    dtype=np.float32
                )

                index = np.array(
                    coordinates[INDEX_TIP],
                    dtype=np.float32
                )

                distance = float(
                    np.linalg.norm(thumb - index)
                )

            # Update robot
            angle = robot.update(distance, dt)

            # Display information
            draw_text(frame, gesture, confidence, coordinates)

            cv2.putText(
                frame,
                f"Robot angle: {angle:.1f} degrees",
                (25, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                HIGHLIGHT,
                2
            )

            cv2.imshow("Hand Gesture Recognition", frame)

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
