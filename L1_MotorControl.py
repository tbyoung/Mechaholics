# SPEED IS EXPRESSED IN RPM
# Import external libraries
import RPi.GPIO as GPIO
import time                                     # only necessary if running this program as a loop
import numpy as np                              # for clip function

#RPi Software PWM Usage: soft_pwm = GPIO.PWM([pin], [freq])
#Software PWM Parameters: Frequency, Duty Cycle, Channel (pin)
GPIO.setmode(GPIO.BCM)
freq = 1000
IN1 = 18
IN2 = 12
IN3 = 13
IN4 = 19
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
R_fwd = GPIO.PWM(IN1, freq)
R_bwd = GPIO.PWM(IN2, freq)
L_fwd = GPIO.PWM(IN3, freq)
L_bwd = GPIO.PWM(IN4, freq)
R_fwd.start(0)
R_bwd.start(0)
L_fwd.start(0)
L_bwd.start(0)

# define functions to command motors, effectively controlling PWM
def MotorL(speed):                              #Speed should be RPM or RAD/S
    if speed < 0:
        L_bwd.ChangeDutyCycle(-speed * 100)
        L_fwd.ChangeDutyCycle(0)
    else:    
        L_fwd.ChangeDutyCycle(speed * 100)
        L_bwd.ChangeDutyCycle(0)


def MotorR(speed):                              #Speed should be RPM or RAD/S
    if speed < 0:
        R_bwd.ChangeDutyCycle(-speed * 100)
        R_fwd.ChangeDutyCycle(0)
    else:
        R_fwd.ChangeDutyCycle(speed * 100)
        R_bwd.ChangeDutyCycle(0)

if __name__ == "__main__":
    while(1):
        u_in = str(input("Enter F or B for fwd or bwd:"))
        if u_in == 'F':
            u_in = int(input("Enter duty cycle (0 - 100), -1 to change pwm frequency (Hz):"))
        
            if 0 <= u_in <= 100:
                R_fwd.ChangeDutyCycle(u_in)
                L_fwd.ChangeDutyCycle(u_in)
    
            else:
                break
        elif u_in == 'B':
            u_in = int(input("Enter duty cycle (0 - 100), -1 to change pwm frequency (Hz):"))
        
            if 0 <= u_in <= 100:
                R_fwd.ChangeDutyCycle(u_in)
                L_bwd.ChangeDutyCycle(u_in)
    
            else:
                break
        else:
            break