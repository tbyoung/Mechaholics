# Import libraries
import RPi.GPIO as GPIO
import time
# Set GPIO numbering mode
GPIO.setmode(GPIO.BCM)

# Set pin 11 as an output, and define as servo1 as PWM pin
GPIO.setup(24,GPIO.OUT)
servo1 = GPIO.PWM(24,50) # pin 11 for servo1, pulse 50Hz

# Loop to allow user to set servo angle. Try/finally allows exit
# with execution of servo.stop and GPIO cleanup :)
def setServoAngle(angle):
    try:
        servo1.ChangeDutyCycle(2+(angle/18)) # 180 for fwd, 105 for right
        time.sleep(0.5)
        servo1.ChangeDutyCycle(0)
    except:
        servo1.stop()
        #Clean things up at the end
        # servo1.stop()

# Start PWM running, with value of 0 (pulse off)
servo1.start(0)

if __name__ == "__main__":
    while 1:
        u_in = int(input("Input servo angle: "))
        try:
            setServoAngle(u_in)
        except:
            print("incorrect input, exiting")
            break
    servo1.stop()
