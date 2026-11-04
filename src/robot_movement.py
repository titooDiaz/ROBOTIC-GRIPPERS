import time
import serial

# DIFFERENTIAL EQUATION SETTINGS
# d(theta)/dt = K * (theta_desired - theta)

K = 5.0
MAX_REFERENCE = 100.0

MIN_DISTANCE_PX = 30.0
MAX_DISTANCE_PX = 250.0

MIN_ANGLE = 0.0
MAX_ANGLE = 180.0

SERIAL_PORT = "/dev/ttyACM0"
#  /dev/ttyUSB0
# /dev/ttyACM0
BAUD_RATE = 115200

SEND_INTERVAL = 0.05


class RobotMovement:
    def __init__(self):
        self.ser = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=0.01
        )

        # The ESP32 may restart when the serial port opens.
        time.sleep(2)
        self.ser.reset_input_buffer()
        self.current_angle = 0.0
        self.last_sent_angle = None
        self.last_send_time = 0.0
        print("Robot connected successfully.")

    def calculate_reference(self, distance):
        """
        Converts finger distance into a reference
        between 0 and MAX_REFERENCE.
        """
        distance = max(MIN_DISTANCE_PX, min(distance, MAX_DISTANCE_PX))
        reference = (
            (distance - MIN_DISTANCE_PX)
            / (MAX_DISTANCE_PX - MIN_DISTANCE_PX)
        ) * MAX_REFERENCE
        return reference

    def calculate_desired_angle(self, reference):
        """
        Linear relationship:
        theta_desired = 180 * reference / MAX_REFERENCE
        """
        return 180.0 * reference / MAX_REFERENCE

    def solve_euler(self, desired_angle, dt):
        """
        Differential equation:
        d(theta)/dt = K * (theta_desired - theta)
        Euler method:
        theta_next = theta + K * error * dt
        """
        error = desired_angle - self.current_angle
        dt = min(dt, 0.05)
        self.current_angle += K * error * dt

        self.current_angle = max(
            MIN_ANGLE,
            min(self.current_angle, MAX_ANGLE)
        )
        return self.current_angle

    def send_angle(self, angle):
        """
        Sends the calculated angle to the ESP32.
        Avoids transmitting tiny changes.
        """
        now = time.monotonic()
        if now - self.last_send_time < SEND_INTERVAL:
            return

        angle = int(round(angle))
        if (self.last_sent_angle is None or abs(angle - self.last_sent_angle) >= 2):
            self.ser.write(f"{angle}\n".encode())
            self.last_sent_angle = angle
            self.last_send_time = now

    def update(self, distance, dt):
        if distance is None:
            return self.current_angle
        reference = self.calculate_reference(distance)
        desired_angle = self.calculate_desired_angle(reference)
        angle = self.solve_euler(desired_angle, dt)
        self.send_angle(angle)
        return angle

    def close(self):
        if self.ser.is_open:
            self.ser.close()
        print("Robot disconnected.")