#GPIO Pin 25 BCM pin type
#Low default for retraction (no modulation as to distance of linear actuator)
#High - Extend at max
#Low - Retract
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT)

def actuate(data=0):
    if data:
        GPIO.output(25, GPIO.HIGH)
    else:
        GPIO.output(25, GPIO.LOW)

if __name__ == "__main__":
    exit()