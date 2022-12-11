import time as t
import L2_TOF_dataConversion as tof
import L1_TOF as L1tof
import L2_PID as pid
import numpy as np
import L2_Inverse_Kinematics as ik
import L2_Kinematics as k
import L1_MotorControl as m
import L2_LiDAR_dataConversion as lidar
import L1_Camera as cam
import L2_ClassifyImageNew as ci

row_count = 0
iter = 0
theta_dot = 0
x_dot = 0.15
l_close = 0.25 # the minimum distance to the left row that is acceptable from LiDAR scan
r_close = -0.25 # the minimum distance to the right tow that is acceptable from LiDAR scan
ground = 0.40 # y measurement for ground distance from LiDAR
crash_prevent = 0 # wayyy to close value to detect and prevent crashing
num_points = 54
lidar_rest_counter = t.time()
time_turn = 0
l_r_turn_count = "F"
timekeep = 0

while 1: 
    try:
        cam.imgTake()
        print(ci.getClassification())
        curr, dist2row, iter = tof.angDist_avg(iter)
        liscan = lidar.lidar_xy(num_points)
        l_dist = dist2row[0]
        r_dist = dist2row[1]
        l_ang = dist2row[2]
        r_ang = dist2row[3]
        closest_l_ground = 100
        closest_r_ground = -100

        for i in range(num_points):
            if liscan[i][1] > 0:
                if liscan[i][0] < ground and liscan[i][0] > 0.1:
                    if (liscan[i][1] < closest_l_ground):
                        closest_l_ground = liscan[i][1] # checks for the closest value of y from liscan that is above ground level (closest left obstacle)
            if liscan[i][1] < 0:
                if liscan[i][0] < ground and liscan[i][0] > 0.1:
                    if (liscan[i][1] > closest_r_ground):
                        closest_r_ground = liscan[i][1] # checks for the closest value of y from liscan that is above ground level (closest right obstacle)
                
        print(closest_l_ground, closest_r_ground, "closest l ground and closest r ground")
        
        if closest_l_ground < l_close:
            x_dot = 0.10
            theta_dot = -0.1
            print("left too close")
            if time_turn == 0:
                time_turn = t.time()
            else:
                pass
            l_r_turn_count = 'R'
        elif closest_r_ground > r_close:
            x_dot = 0.10
            theta_dot = 0.1
            print("Right too close")
            if time_turn == 0:
                time_turn = t.time()
            else:
                pass
            l_r_turn_count = "L"
        elif time_turn != 0:
            if timekeep == 0:
                time_after_turn = t.time()
                timekeep = time_turn - time_after_turn
            if (t.time() - time_after_turn) < timekeep:
                if l_r_turn_count == 'L':
                    x_dot = 0.10
                    theta_dot = -0.1
                if l_r_turn_count == 'R':
                    x_dot = 0.10
                    theta_dot = 0
            else:
                timekeep = 0
                time_after_turn = 0
                theta_dot = 0
                x_dot = 0.12
        else:
            x_dot = 0.12
            if theta_dot >0:
                theta_dot -= 0.003
            elif theta_dot < 0:
                theta_dot += 0.003
            else:
                theta_dot = 0
            time = 0
            print("Driving straight")
                




        #dist_diff = dist2row[0] - dist2row[1] # left - right
        #if abs(dist_diff) > 100:
        #    theta_dot = np.sign(dist_diff) * 0.1
        #    x_dot = 0.10
        #if x_dot < 0.08:
        #    x_dot = 0.10
        
        #t.sleep(0.5)


        #print("left row dist =", l_dist, "right row dist =", r_dist)
        #print("left row theta =", l_ang, "right row theta =", r_ang)
        print("x_dot =", x_dot, "theta_dot =", theta_dot)
    
        to_convert = [np.round(x_dot, 3), np.round(theta_dot, 3)]
        pdt = ik.convert(to_convert)
        pdc = k.getPdCurrent()
        print(pdt, pdc, "->pdt, pdc")
        pid.driveClosedLoop(pdt, pdc)

    except KeyboardInterrupt:
        m.MotorL(0)
        m.MotorR(0)
        L1tof.cleanup()
        print('Exiting on Keyboard Interrupt.')
        break
