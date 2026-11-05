
#include <Arduino.h>
#include <ESP32Servo.h>

Servo gripperServo;
Servo verticalServo;


// SERVO PINS
const int GRIPPER_PIN = 17;
const int VERTICAL_PIN = 16;


// MOVEMENT SETTINGS
const int STEP_SIZE = 2;
const unsigned long STEP_INTERVAL = 15;


// GRIPPER
int gripperCurrent = 0;
int gripperTarget = 0;


// VERTICAL
int verticalCurrent = 0;
int verticalTarget = 0;


// TIMING
unsigned long previousStep = 0;


// SERIAL BUFFER
String inputBuffer = "";


void processSerial() {

    while (Serial.available() > 0) {

        char received = Serial.read();

        if (received == '\n') {

            inputBuffer.trim();

            int separator = inputBuffer.indexOf(',');

            if (separator > 0) {

                int gripper = inputBuffer.substring(
                    0, separator
                ).toInt();

                int vertical = inputBuffer.substring(
                    separator + 1
                ).toInt();

                if (
                    gripper >= 0 && gripper <= 180 &&
                    vertical >= 0 && vertical <= 150
                ) {

                    gripperTarget = gripper;
                    verticalTarget = vertical;

                }
            }

            inputBuffer = "";

        } else if (received != '\r') {

            if (inputBuffer.length() < 20) {
                inputBuffer += received;
            } else {
                inputBuffer = "";
            }
        }
    }
}


void updateServo(
    Servo &servo,
    int &currentAngle,
    int targetAngle
) {

    if (currentAngle < targetAngle) {

        currentAngle += STEP_SIZE;

        if (currentAngle > targetAngle) {
            currentAngle = targetAngle;
        }

        servo.write(currentAngle);
    }

    else if (currentAngle > targetAngle) {

        currentAngle -= STEP_SIZE;

        if (currentAngle < targetAngle) {
            currentAngle = targetAngle;
        }

        servo.write(currentAngle);
    }
}


void moveServos() {

    unsigned long now = millis();

    if (now - previousStep >= STEP_INTERVAL) {

        previousStep = now;

        updateServo(
            gripperServo,
            gripperCurrent,
            gripperTarget
        );

        updateServo(
            verticalServo,
            verticalCurrent,
            verticalTarget
        );
    }
}


void setup() {

    Serial.begin(115200);

    gripperServo.setPeriodHertz(50);
    verticalServo.setPeriodHertz(50);

    gripperServo.attach(GRIPPER_PIN, 500, 2400);
    verticalServo.attach(VERTICAL_PIN, 500, 2400);

    gripperServo.write(0);
    verticalServo.write(0);

    Serial.println("Robot initialized.");
}


void loop() {
    processSerial();

    moveServos();
}
