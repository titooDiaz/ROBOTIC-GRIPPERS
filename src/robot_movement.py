import time
import serial


# SERIAL SETTINGS
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
SEND_INTERVAL = 0.05


# GRIPPER SETTINGS
K_GRIPPER = 5.0

MIN_DISTANCE_PX = 30.0
MAX_DISTANCE_PX = 250.0

GRIPPER_MIN_ANGLE = 0.0
GRIPPER_MAX_ANGLE = 180.0


# VERTICAL MOVEMENT SETTINGS
K_VERTICAL = 5.0

VERTICAL_MIN_ANGLE = 0.0
VERTICAL_MAX_ANGLE = 150.0

SCREEN_MARGIN = 50


class RobotMovement:

    def __init__(self):

        self.ser = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=0.01
        )

        time.sleep(2)
        self.ser.reset_input_buffer()

        # Current angles
        self.gripper_angle = 0.0
        self.vertical_angle = 0.0

        # Last transmitted angles
        self.last_gripper_sent = None
        self.last_vertical_sent = None

        self.last_send_time = 0.0

        print("Robot connected successfully.")

    # ==================================
    # GRIPPER
    # ==================================

    def calculate_gripper_target(self, distance):

        distance = max(
            MIN_DISTANCE_PX,
            min(distance, MAX_DISTANCE_PX)
        )

        reference = (
            (distance - MIN_DISTANCE_PX)
            / (MAX_DISTANCE_PX - MIN_DISTANCE_PX)
        )

        return 180.0 * reference

    # ==================================
    # VERTICAL MOVEMENT
    # ==================================

    def calculate_vertical_target(self, hand_y, screen_height):

        y_min = SCREEN_MARGIN
        y_max = screen_height - SCREEN_MARGIN

        # Keep the hand position inside the useful screen area
        hand_y = max(y_min, min(hand_y, y_max))

        # Top = 150 degrees
        # Bottom = 0 degrees

        reference = (y_max - hand_y) / (y_max - y_min)

        return VERTICAL_MAX_ANGLE * reference

    # ==================================
    # EULER METHOD
    # ==================================

    def solve_euler(self, current_angle, desired_angle, k, dt):

        dt = min(max(dt, 0.0), 0.05)

        error = desired_angle - current_angle

        next_angle = current_angle + k * error * dt

        return max(0.0, min(next_angle, 180.0))

    # ==================================
    # SERIAL COMMUNICATION
    # ==================================

    def send_angles(self, gripper_angle, vertical_angle):

        now = time.monotonic()

        if now - self.last_send_time < SEND_INTERVAL:
            return

        gripper = int(round(gripper_angle))
        vertical = int(round(vertical_angle))

        gripper_changed = (
            self.last_gripper_sent is None
            or abs(gripper - self.last_gripper_sent) >= 2
        )

        vertical_changed = (
            self.last_vertical_sent is None
            or abs(vertical - self.last_vertical_sent) >= 2
        )

        if gripper_changed or vertical_changed:

            message = f"{gripper},{vertical}\n"

            self.ser.write(message.encode())

            self.last_gripper_sent = gripper
            self.last_vertical_sent = vertical

            self.last_send_time = now

    # ==================================
    # UPDATE BOTH MOTORS
    # ==================================

    def update(self, distance, hand_y, screen_height, dt):

        # Keep the previous positions if no hand is detected
        if distance is None or hand_y is None:
            return self.gripper_angle, self.vertical_angle

        # Calculate desired angles
        desired_gripper = self.calculate_gripper_target(distance)

        desired_vertical = self.calculate_vertical_target(
            hand_y,
            screen_height
        )

        # Solve both differential equations using Euler
        self.gripper_angle = self.solve_euler(
            self.gripper_angle,
            desired_gripper,
            K_GRIPPER,
            dt
        )

        self.vertical_angle = self.solve_euler(
            self.vertical_angle,
            desired_vertical,
            K_VERTICAL,
            dt
        )

        # Send both angles to ESP32
        self.send_angles(
            self.gripper_angle,
            self.vertical_angle
        )

        return self.gripper_angle, self.vertical_angle

    def close(self):

        if self.ser.is_open:
            self.ser.close()

        print("Robot disconnected.")