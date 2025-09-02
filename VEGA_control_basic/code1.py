from gpiozero import PWMOutputDevice, DistanceSensor, AngularServo
from gpiozero.pins.lgpio import LGPIOFactory
import time
from time import sleep
from transitions import Machine

# Set GPIO mode (BCM for sensors, BOARD for motors)
# GPIO.setmode(GPIO.BOARD)

factory = LGPIOFactory()

# Motor A (Left) and Motor B (Right) Configuration
# ENA = PWMOutputDevice(12, pin_factory=factory)  # Left Motor PWM
INT1 = PWMOutputDevice(12, pin_factory=factory)   # Left Motor Input 1
INT2 = PWMOutputDevice(16, pin_factory=factory)   # Left Motor Input 2
# ENB = PWMOutputDevice(13, pin_factory=factory)  # Right Motor PWM
INT3 = PWMOutputDevice(13, pin_factory=factory)   # Right Motor Input 3
INT4 = PWMOutputDevice(5, pin_factory=factory)    # Right Motor Input 4

# Ultrasonic Sensor (BCM Mode)
ultrasonic = DistanceSensor(echo=21, trigger=23, max_distance=2.0, pin_factory=factory)

# Servo Motor (BCM Mode)
servo = AngularServo(18, min_angle=0, max_angle=180, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000, pin_factory=factory)
servo.angle = 90  # Default to front

# Define States for State Machine
states = ["Idle", "Front", "Back", "Left", "Right"]

class CarStateMachine:
    def __init__(self):
        self.machine = Machine(model=self, states=states, initial="Idle")
        self.machine.add_transition("Start", "Idle", "Front")
        self.machine.add_transition("Reverse", "Front", "Back")
        self.machine.add_transition("Turn_Left", "Front", "Left")
        self.machine.add_transition("Turn_Right", "Front", "Right")
        self.speed = 0.5  # Default speed

    def move_forward(self, speed=0.9):
        INT1.value = speed
        INT2.value = 0
        INT3.value = speed
        INT4.value = 0
        # ENA.value = 0.5
        # ENB.value = 0.5
        print(f"Moving forward at {speed}% speed.")

    def move_backward(self, speed=0.9):
        INT1.value = 0
        INT2.value = speed
        INT3.value = 0
        INT4.value = speed
        # ENA.value = 0.5
        # ENB.value = 0.5
        print(f"Moving backward at {speed}% speed.")

    def turn_left(self, speed=0.2):
        INT1.value = speed  # Reduce left motor speed
        INT3.value = speed * 1.8  # Keep right motor speed high
        print(f"Turning left: Left Motor {speed * 0.5}%, Right Motor {speed}%.")

    def turn_right(self, speed=0.2):
        INT1.value = speed * 1.8  # Keep left motor speed high
        INT3.value = speed  # Reduce right motor speed
        print(f"Turning right: Left Motor {speed}%, Right Motor {speed * 0.5}%.")

    def stop_movement(self):
        INT1.value = 0
        INT2.value = 0
        INT3.value = 0
        INT4.value = 0
        print("Car stopped.")

def get_distance():
    """Reads and returns distance in centimeters, with retries if no echo is received."""
    for _ in range(3):  # Retry up to 3 times
        distnce = round(ultrasonic.distance * 100, 2)
        if distance > 0:
            print(f"Current Distance: {distance} cm")
            return distance
        sleep(0.1)  # Small delay before retry
    print("Ultrasonic sensor error: no valid reading.")
    return 100  # Default to safe value

def move_servo(angle):
    """Moves servo to specified angle (0-180)"""
    if 0 <= angle <= 180:
        servo.angle = angle
        sleep(0.5)
    else:
        print("Invalid angle! Must be between 0 and 180.")

def auto_mode():
    """Automated mode for obstacle avoidance."""
    car = CarStateMachine()

    while True:
        command = input("Enter command (f=start, s=stop, q=quit): ").strip().lower()
        if command == 'f':
            print("Starting auto mode...")
            car.move_forward()

            while True:
                distance = get_distance()

                if distance < 30:  # Obstacle detected
                    print("Obstacle detected! Stopping car.")
                    car.stop_movement()
                    sleep(1)

                    # Scan left
                    move_servo(140)
                    sleep(1)
                    left_distance = get_distance()

                    # Scan right
                    move_servo(40)
                    sleep(1)
                    right_distance = get_distance()

                    # Reset to forward
                    move_servo(90)
                    sleep(1)

                    if left_distance > 30 and right_distance <= 30:
                        print("Turning left...")
                        car.move_forward()
                        car.turn_left(40)
                        sleep(2)

                    elif right_distance > 30 and left_distance <= 30:
                        print("Turning right...")
                        car.move_forward()
                        car.turn_right(40)
                        sleep(2)

                    elif left_distance > 30 and right_distance > 30:
                        user_choice = input("Both sides clear. Turn left (l) or right (r)? ").strip().lower()
                        if user_choice == 'l':
                            car.move_forward()
                            car.turn_left(40)
                            sleep(2)
                        elif user_choice == 'r':
                            car.move_forward()
                            car.turn_right(40)
                            sleep(2)

                sleep(0.5)  # Maintain 1-second update cycle

        elif command == 's':
            car.stop_movement()
        elif command == 'q':
            print("Exiting auto mode.")
            break
        else:
            print("Invalid command.")

def manual_mode():
    """Manual control mode."""
    car = CarStateMachine()

    while True:
        command = input("Enter command (f=forward, b=backward, l=left, r=right, s=stop, q=quit): ").strip().lower()

        if command == 'f':
            car.move_forward()
        elif command == 'b':
            car.move_backward()
        elif command == 'l':
            car.turn_left()
        elif command == 'r':
            car.turn_right()
        elif command == 's':
            car.stop_movement()
        elif command == 'q':
            print("Exiting manual mode.")
            break
        else:
            print("Invalid command.")

try:
    mode = input("Select mode (m=manual, a=auto): ").strip().lower()
    if mode == 'm':
        manual_mode()
    elif mode == 'a':
        auto_mode()
    else:
        print("Invalid selection. Exiting.")
except KeyboardInterrupt:
    print("\nProgram interrupted.")
finally:
    print("Program ended. No GPIO cleanup needed for gpiozero-controlled devices.")
