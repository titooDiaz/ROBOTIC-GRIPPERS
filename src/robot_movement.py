import time
import serial

import settings


class RobotMovement:

    def __init__(self):

        self.ser = serial.Serial(
            settings.SERIAL_PORT,
            settings.BAUD_RATE,
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
            settings.MIN_DISTANCE_PX,
            min(distance, settings.MAX_DISTANCE_PX)
        )

        reference = (
            (distance - settings.MIN_DISTANCE_PX)
            / (settings.MAX_DISTANCE_PX - settings.MIN_DISTANCE_PX)
        )

        angle_range = (
            settings.GRIPPER_MAX_ANGLE - settings.GRIPPER_MIN_ANGLE
        )
        return settings.GRIPPER_MIN_ANGLE + angle_range * reference

    # ==================================
    # VERTICAL MOVEMENT
    # ==================================

    def calculate_vertical_target(self, hand_y, screen_height):

        y_min = settings.SCREEN_MARGIN
        y_max = screen_height - settings.SCREEN_MARGIN

        # Keep the hand position inside the useful screen area
        hand_y = max(y_min, min(hand_y, y_max))

        reference = (y_max - hand_y) / (y_max - y_min)

        angle_range = (
            settings.VERTICAL_MAX_ANGLE - settings.VERTICAL_MIN_ANGLE
        )
        return settings.VERTICAL_MIN_ANGLE + angle_range * reference

    # ==================================
    # EULER METHOD
    # ==================================

    def solve_euler(
        self,
        current_angle,
        desired_angle,
        k,
        dt,
        min_angle,
        max_angle,
    ):

        dt = min(max(dt, 0.0), settings.MAX_EULER_DT)

        error = desired_angle - current_angle

        next_angle = current_angle + k * error * dt

        return max(
            min_angle,
            min(next_angle, max_angle),
        )

    # ==================================
    # SERIAL COMMUNICATION
    # ==================================

    def send_angles(self, gripper_angle, vertical_angle):

        now = time.monotonic()

        if now - self.last_send_time < settings.SEND_INTERVAL:
            return

        gripper = int(round(gripper_angle))
        vertical = int(round(vertical_angle))

        gripper_changed = (
            self.last_gripper_sent is None
            or abs(gripper - self.last_gripper_sent)
            >= settings.SERVO_CHANGE_THRESHOLD
        )

        vertical_changed = (
            self.last_vertical_sent is None
            or abs(vertical - self.last_vertical_sent)
            >= settings.SERVO_CHANGE_THRESHOLD
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
            settings.K_GRIPPER,
            dt,
            settings.GRIPPER_MIN_ANGLE,
            settings.GRIPPER_MAX_ANGLE,
        )

        self.vertical_angle = self.solve_euler(
            self.vertical_angle,
            desired_vertical,
            settings.K_VERTICAL,
            dt,
            settings.VERTICAL_MIN_ANGLE,
            settings.VERTICAL_MAX_ANGLE,
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