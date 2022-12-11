import smbus2
import numpy as np
import time

bus = smbus2.SMBus(1)

encL = 0x41
encR = 0x40

def encRead(encoder):
    try:
        encoderBytes = bus.read_i2c_block_data(encoder, 0xFE, 2)
        usefulBits = (encoderBytes[0] << 6) | encoderBytes[1]
        degreesPos = usefulBits*(360/2**14) # turn angle into degrees
        degreesPos = round(degreesPos, 1) # round to xxx.x 
    except:
        print("Encoder Reading Failed.")
        degreesPos = 0
    return degreesPos

def readShaftPositions():
    try:
        rawAngle = encRead(encL)
        angleL = 360.0 - rawAngle
        angleL = round(angleL, 1)
    except:
        print('Could not read left encoder.')
        angleL = 0

    try:
        angleR = encRead(encR)
        
    except:
        print('Could not read right encoder.')
        angleR = 0
    angles = np.array([angleL, angleR])
    return angles

if __name__ == "__main__":
    while 1:
        try:
            print(readShaftPositions())

        except:
            print("Exiting")
            break

