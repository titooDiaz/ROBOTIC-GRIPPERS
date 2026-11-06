"""Project settings.

Edit the values in this file to adjust the training and robot behavior.
"""

from pathlib import Path


# Folder where trained models and scalers are saved.
MODEL_DIR = Path("models")

# Path to the trained gesture model.
MODEL_PATH = MODEL_DIR / "gesture_model.pth"

# Path to the feature scaler used by the model.
SCALER_PATH = MODEL_DIR / "scaler.joblib"

# Minimum confidence required to accept a predicted gesture.
CONFIDENCE_THRESHOLD = 0.60

# Normal landmark color in the camera window (BGR).
LIGHT = (220, 220, 220)

# Highlight color in the camera window (BGR).
HIGHLIGHT = (0, 220, 255)

# Landmark index for the thumb tip.
THUMB_TIP = 4

# Landmark index for the index finger tip.
INDEX_TIP = 8

# Landmark index for the center of the palm.
PALM_CENTER = 9

# Empty border around the useful vertical screen area, in pixels.
SCREEN_MARGIN = 50

# Number of training passes over the dataset.
EPOCHS = 100

# Step size h in the Euler method.
LEARNING_RATE = 0.001

# Epochs without improvement before training stops.
PATIENCE = 12

# Serial device used by the ESP32.
SERIAL_PORT = "/dev/ttyACM0"

# Serial communication speed.
BAUD_RATE = 115200

# Minimum time between serial messages, in seconds.
SEND_INTERVAL = 0.05

# Minimum angle change required before sending a new value.
SERVO_CHANGE_THRESHOLD = 2

# Maximum time step used by the Euler method, in seconds.
MAX_EULER_DT = 0.05

# Camera device index used by OpenCV.
CAMERA_INDEX = 0

# Gripper control gain for the Euler method.
K_GRIPPER = 5.0

# Minimum finger distance used for gripper control, in pixels.
MIN_DISTANCE_PX = 30.0

# Maximum finger distance used for gripper control, in pixels.
MAX_DISTANCE_PX = 250.0

# Minimum gripper angle, in degrees.
GRIPPER_MIN_ANGLE = 0.0

# Maximum gripper angle, in degrees.
GRIPPER_MAX_ANGLE = 180.0

# Vertical control gain for the Euler method.
K_VERTICAL = 5.0

# Minimum vertical angle, in degrees.
VERTICAL_MIN_ANGLE = 0.0

# Maximum vertical angle, in degrees.
VERTICAL_MAX_ANGLE = 150.0
