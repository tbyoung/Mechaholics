# This program takes the encoder values from encoders, computes wheel movement
# This program was initially published for the Scuttle robotics platform, and has been modified for use by the AMAR Robot

import L1_Encoder as enc                    # local library for encoders
import numpy as np                          # library for math operations
import time                                 # library for time access

# define kinematics
R = 0.022                                # wheel radius
L = 0.110                                     # half of the wheelbase
res = (360/2**14)                           # resolution of the encoders (deg)
roll = int(360/res)                         # variable for rollover logic
gap = 0.5*roll                            # degress specified as limit for rollover

A = np.array([[R/2, R/2], [-R/(2*L), R/(2*L)]])     # This matrix relates [PDL, PDR] to [XD,TD]

wait = 0.02                                 # wait time between encoder measurements (s)


def getTravel(deg0, deg1):                  # calculate the delta on Left wheel
    trav = deg1 - deg0                      # reset the travel reading
    #print("Travel: ",trav)
    if((-trav) >= gap):                     # if movement is large (has rollover)
        trav = (deg1 - deg0 + roll)         # forward rollover
    if(trav >= gap):
        trav = (deg1 - deg0 - roll)         # reverse rollover
    return(trav)


# Note:  this function takes at least 5ms to run.  It also populates a global
# variable so programs can access the previous measurement instantaneously.
def getPdCurrent():
    global pdCurrents                       # make a global var for easy retrieval
    encoders_t1 = enc.readShaftPositions()                   # grabs the current encoder readings in degrees
    t1 = time.monotonic()                        # time.time() reports in seconds
    time.sleep(wait)
    encoders_t2 = enc.readShaftPositions()                   # grabs the current encoder readings in degrees
    t2 = time.monotonic()
    global deltaT
    deltaT = round((t2 - t1), 3)            # new scalar dt value

    # ---- movement calculations
    travel = np.array(encoders_t2 - encoders_t1)             # this wheel is inverted from the right side
    travel_b = np.array(travel + 360)
    travel_c = np.array(travel - 360)
    mx = np.array([travel, travel_b, travel_c])
    mx = np.absolute(mx)
    mins = np.argmin(mx, 0)
    left = mx[mins[0], 0]
    right = mx[mins[1], 1]
    wheelTravel = np.array([left, right])
    
    wheelSpeeds_deg = wheelTravel / deltaT
    pdCurrents = wheelSpeeds_deg * np.pi / 180
    return (pdCurrents) # Current pdl, pdr in rad/s

def phiTravels(encoders_t1, encoders_t2):   # get travel of wheels [deg, deg] (take no measurements)
    travel = encoders_t2 - encoders_t1      # compute change in both shaft encoders (degrees)
    travel = encoders_t2 - encoders_t1      # array, 2x1 to indicate travel
    trav_b = travel + 360                   # array variant b
    trav_c = travel - 360                   # array variant c
    mx = np.stack((travel, trav_b, trav_c)) # combine array variants
    mx_abs = np.absolute(mx)                # convert to absolute val
    mins = np.argmin(mx_abs,0)              # find the indices of minimum values (left and right hand)
    left = mx[mins[0],0]                    # pull corresponding indices from original array
    right = mx[mins[1],1]                   # pull corresponding index for RH
    wheelTravel = np.array([left,right])    # combine left and right sides to describe travel (degrees)
    return(wheelTravel)    



def getMotion():                            # this function returns the chassis speeds
    B = getPdCurrent()                      # store phidots to array B (here still in rad/s)
    C = np.matmul(A, B)                     # perform matrix multiplication
    C = np.round(C, decimals=3)             # round the matrix
    return(C)                               # returns a matrix containing [xDot, thetaDot]

# THIS SECTION ONLY RUNS IF THE PROGRAM IS CALLED DIRECTLY
if __name__ == "__main__":
    while True:
        C = getMotion()
        print("xdot(m/s), thetadot (rad/s):", C)
        time.sleep(0.1)