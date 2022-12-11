import time as t
import L2_PID as pid # print statements removed
import numpy as np
import L2_Inverse_Kinematics as ik # print statements removed
import L2_Kinematics as k # print statements removed
import L1_MotorControl as m # print statements removed
import L2_LiDAR_dataConversion as lidar # print statements removed
import L2_ClassifyImage as classify
import L1_ServoControl as serv
import L2_WeedWhack as ww
#import L1_Camera as cam
import L1_IMU as imu
import os



crash_prevent = 0.08 # When obstacles are too close to the
singlerow_distance = 0 # initializing single-row follower variable
singlerow_indicator = "" # initializing single-row follower variable
next_row_turn = "L" # initially turns to the right when no rows detected.
turn_distance_from_row = 0.35 # distance the robot should stay from the row it is turning with resepct to. (end of row distance)
no_row_timer = 0 # timer for no rows present condition, to ensure no row switching mid row
one_row_timer = 0
iter_count = 0
init_img_find_count = 1
no_print_spam = 0


def sendToPID(x_dot, theta_dot):
    to_convert = [np.round(x_dot, 3), np.round(theta_dot, 3)]
    pdt = ik.convert(to_convert)
    pdc = k.getPdCurrent()
    pid.driveClosedLoop(pdt, pdc)

def fullstop():
    timer = t.time()
    while (t.time() - timer) < 0.5:
        m.MotorL(0)
        m.MotorR(0)
    pid.resetIntegralTerm()
    return

serv.setServoAngle(110)
while 1: 
    try:
        iter_count += 1
        if iter_count >= 15:
            #weedDetect = classify.getClassification()
            weedDetect = ['not a weed', 0.50]
            print(weedDetect)
            
            if(weedDetect[0][0]=='pigweed' and weedDetect[0][1] > .70):
                angleTrack = 110
                #start = t.time()
                fullstop()
                os.rename("/home/pi/mechaholics/img.jpg", ("/home/pi/mechaholics/stop_img_" + str(init_img_find_count) + ".jpg"))
                init_img_find_count += 1
                start = t.time()
                stop = t.time()
                aquired = 0
                serv.setServoAngle(110)
                ww_loop_iter = 0
                ww_timeKeep = 0
                while angleTrack < 180:
                    angleTrack += 5
                    if angleTrack > 180:
                        angleTrack = 180
                    serv.setServoAngle(angleTrack)
                    weedDetect = classify.getClassification()
                    iter_count = 0
                    while (weedDetect[0][0] != 'pigweed'and weedDetect[0][1]<.70 and ww_loop_iter < 100):
                        start = t.time()
                        start2 = t.time()
                        stop2 = t.time()
                        while stop2-start2 < 0.25:
                            sendToPID(0, -0.15)
                        fullstop()
                        if iter_count == 7:
                            weedDetect = classify.getClassification()
                            iter_count = 0
                        stop = t.time()
                        ww_timeKeep = ww_timeKeep + (stop-start)
                        iter_count += 1
                        ww_loop_iter += 1
                        #stop = t.time()
                    if ww_loop_iter >= 100:
                        break
                stop = t.time()
                fullstop()
                weedDetect = classify.getClassification()               
                if(weedDetect[0][0]=='pigweed' and weedDetect[0][1] > 0.70):
                    #ww.engage()
                    print("WEED WHACK!!!!!")
                start = t.time()
                while start - stop < ww_timeKeep*2:
                    sendToPID(0, 0.04)
                    stop = t.time()
                #start = t.time()
                fullstop()
                weedDetect = classify.getClassification()
                serv.setServoAngle(110)


                

            iter_count = 0
        closest_L_obstacle, closest_R_obstacle = lidar.find_closest_obstacles()
        
        if (closest_L_obstacle < crash_prevent) or (closest_R_obstacle > -crash_prevent): # Checking for imminent crash and averting course
            if no_print_spam != 1:
                print("Imminent crash detected!")
                no_print_spam = 1
            print(closest_L_obstacle, closest_R_obstacle)
            fullstop()
            if closest_L_obstacle > abs(closest_R_obstacle):
                x_dot = -0.05
                theta_dot = 0.15
            elif closest_L_obstacle < abs(closest_R_obstacle):
                x_dot = -0.05
                theta_dot = -0.15
            stop_timer = t.time()
            while (t.time() - stop_timer) < 1:
                sendToPID(x_dot, theta_dot)
            fullstop()
            x_dot = 0
            theta_dot = 0
        elif (closest_L_obstacle < 1.0) and (closest_R_obstacle > -1.0): # Checking if both rows are present in scan
            
            singlerow_distance = 0 # resetting singlerow distance when both rows are present again
            no_row_timer = 0 # resetting no_row_timer if no row was detected on previous iteration
            one_row_timer = 0
            singlerow_indicator = ''
            if -0.025 <= (closest_L_obstacle + closest_R_obstacle) <= 0.025: # if the distance between the rows is in an acceptable tolerance
                if no_print_spam != 2:
                    print("Both rows present, driving straight.")
                    no_print_spam = 2
                x_dot = 0.12
                theta_dot = 0
            else:
                if no_print_spam != 3:
                    print("Both rows present, adjusting for center.")
                    no_print_spam = 3
                theta_dot = (closest_L_obstacle - abs(closest_R_obstacle))/2 #0.08
                x_dot = 0.1
        elif ((closest_L_obstacle > 1.0) and (closest_R_obstacle > -1.0)) or (singlerow_indicator == 'R'): # If the left row is no longer scannable, follows right row
            no_row_timer = 0 # resetting no_row_timer if no row was detected on previous iteration
            if one_row_timer == 0:
                one_row_timer = t.time() # sets the current time as the most recent instance of one row, to ensure small objects don't interfere with
                                         # singlerow navigation.
            if (singlerow_distance == 0) or (singlerow_indicator != 'R'): # setting singlerow distance and indicator for left row following
                singlerow_distance = closest_R_obstacle
                singlerow_indicator = 'R'
            if (t.time() - one_row_timer) < 1.5:
                x_dot = 0.1
                theta_dot = 0
                if no_print_spam != 4:
                    print("Left row missing. Driving forwards for 1 second.")
                    no_print_spam = 4
            elif closest_R_obstacle != singlerow_distance: # turn left when row gets closer
                if no_print_spam != 5:
                    print('Left row missing. Adjusting distance to right row.')
                    no_print_spam = 5
                x_dot = 0.1
                theta_dot = (closest_R_obstacle - singlerow_distance)/2
            else: # this condition will likely never happen
                if no_print_spam != 5:
                    print("Left row in correct spot. Driving forwards.")
                    no_print_spam = 5
                x_dot = 0.1
                theta_dot = 0
        elif ((closest_L_obstacle < 1.0) and (closest_R_obstacle < -1.0)) or (singlerow_indicator == 'L'): # If the right row is no longer scannable, do this 
            no_row_timer = 0 # resetting no_row_timer if no row was detected on previous iteration
            if one_row_timer == 0:
                one_row_timer = t.time()
            if (t.time() - one_row_timer) < 1.5:
                x_dot = 0.1
                theta_dot = 0
                if no_print_spam != 6:
                    print("Right row missing. Driving forwards for 1 second.")
                    no_print_spam = 6
            
            elif (singlerow_distance == 0) or (singlerow_indicator != 'L'): # setting singlerow distanace and indicator for right row following
                singlerow_distance = closest_L_obstacle
                singlerow_indicator = 'L'
            elif closest_L_obstacle != singlerow_distance: # turn right when row gets closer
                print("Right row missing. Adjusting distance to left row.")
                x_dot = 0.1
                theta_dot = -(singlerow_distance - closest_L_obstacle)/2
            else: # this condition will likely never happen
                if no_print_spam != 7:
                    print("Right row in correct spot. Driving forwards.")
                    no_print_spam = 7
                x_dot = 0.1
                theta_dot = 0
        else: # neither row is present, give a couple seconds to see if it will reappear. If not, then switch rows.
            if no_row_timer == 0: # counts the time in which both rows have dissappeared.
                no_row_timer = t.time()
            no_row_timer_calc = t.time() - no_row_timer
            if no_row_timer_calc < 1: # If no rows detected after 2 seconds, robot will attempt to switch rows.
                if no_print_spam != 8:
                    print("No rows detected, waiting 1 second before switching rows.")
                    no_print_spam = 8
                x_dot = 0.1
                theta_dot = 0
            else:
                print("Neither row present. Switching rows to the ", end='')
                fullstop()
                if next_row_turn == "R":
                    print("right.")
                    #while (closest_L_obstacle > 1.0) or (closest_R_obstacle < -1.0):
                    #    closest_L_obstacle, closest_R_obstacle = lidar.find_closest_obstacles() # reacquire scans in the while loop
                    #    x_dot = 0
                    #    theta_dot = -0.15 
                    #    sendToPID(x_dot, theta_dot)
                    rotation_array = []
                    start_timer = t.time()
                    degrees_per_second = 0
                    while (degrees_per_second < 0.9):
                       imu_data = imu.getIMUReadings()
                       rotation_array.append(imu_data)
                       degrees_per_second = (t.time() - start_timer) * np.mean(rotation_array)
                       print(degrees_per_second)
                       m.MotorL(0.5)
                       m.MotorR(0.25)
                    fullstop()
                    rotation_array = []
                    start_timer = t.time()
                    degrees_per_second = 0
                    print('performing second turn.')
                    while (degrees_per_second < 90):
                        imu_data = imu.getIMUReadings()
                        rotation_array.append(imu_data)
                        degrees_per_second = (t.time() - start_timer) * np.mean(rotation_array)
                        print(degrees_per_second)
                        m.MotorL(-0.25)
                        m.MotorR(-0.5)
                    fullstop()
                    closest_L_obstacle, closest_R_obstacle = lidar.find_closest_obstacles()
                    while (closest_L_obstacle > 1.0) or (closest_R_obstacle < -1.0):
                        closest_L_obstacle, closest_R_obstacle = lidar.find_closest_obstacles() # reacquire scans in the while loop
                        x_dot = 0.08
                        theta_dot = 0
                        sendToPID(x_dot, theta_dot)

                    next_row_turn = "L" # set next row turn after while loop closes (left side is within tolerance)
                    print("New row acquired.")
                    #fullstop()
                    #no_row_timer = t.time()
                elif next_row_turn == "L":
                    print("left.")

                    while (closest_R_obstacle < -1.0):

                        closest_L_obstacle, closest_R_obstacle = lidar.find_closest_obstacles() # reacquire scans in the while loop

                        if closest_L_obstacle > turn_distance_from_row:
                            x_dot = 0.06
                            theta_dot = 0.12
                        elif closest_L_obstacle < turn_distance_from_row:
                            x_dot = 0.08
                            theta_dot = 0
                        
                        sendToPID(x_dot, theta_dot)

                    next_row_turn = "R" # set next row turn after while loop closes (right side is within tolerance)
                    print("row reacquired.")
                    fullstop()
                    no_row_timer = t.time()
                
        if abs(theta_dot) > 0.1:
            theta_dot = np.sign(theta_dot) * 0.1
        sendToPID(x_dot, theta_dot)

    except KeyboardInterrupt:
        fullstop()
        print('Exiting on Keyboard Interrupt.')
        break

