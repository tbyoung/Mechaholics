import time as t
import L2_TOF_dataConversion as tof
import L1_TOF as L1tof
import L2_PID as pid
import numpy as np
import L2_Inverse_Kinematics as ik
import L2_Kinematics as k
import L1_MotorControl as m
import csv

row_count = 0
iter = 0
theta_dot = 0
x_dot = 0.15
filename = "nav_test.csv"
header = ['left_dist', 'right_dist', 'left_theta', 'right_theta', 'x_dot', 'theta_dot']
file = open(filename, 'w', newline='')
csvwrite = csv.writer(file)
csvwrite.writerow(header)

while 1: 
    try:
        x_dot = 0.12
        start_time = t.time()
        curr, dist2row, iter = tof.angDist_avg(iter) # [0] is l_dist, [1] is r_dist, [2] is l_theta, [3] is r_theta
        # max x_dot is 0.1
#        ang_diff = dist2row[1] - dist2row[0]
#        theta_dot = dist_diff / 80 + np.radians(dist2row[3])
#        if abs(theta_dot) >= 0.5:
#            sign = np.sign(theta_dot)
#            theta_dot = 0.5 * sign 
        #if dist2row[2] < 5 and dist2row[3] > 5:
        #ang_diff = dist2row[2] - dist2row[3]
        #if ang_diff > 30:
        #    if theta_dot < 0:
        #        theta_dot = 0
        #    theta_dot = theta_dot + np.radians(abs(ang_diff / 6))
            # x_dot -= 0.001
        #elif dist2row[2] > 5 and dist2row[3] < 5:
        #elif ang_diff < 30:
        #    if theta_dot > 0:
        #        theta_dot = 0
        #    theta_dot = theta_dot - np.radians(abs(ang_diff / 6))
            # x_dot -= 0.001
        #else:
        #    theta_dot = 0
        #    x_dot = 0.15
        dist_diff = dist2row[0] - dist2row[1] # left - right
        if dist_diff > 40:
            theta_dot -= 0.006
            x_dot = 0.10
        elif dist_diff < 40:
            theta_dot += 0.006
            x_dot = 0.10
        if abs(dist_diff) > 100:
            theta_dot = np.sign(dist_diff) * 0.08
            x_dot = 0.10
        if abs(theta_dot) > 0.08:
            theta_dot = np.sign(theta_dot) * 0.08
        if x_dot < 0.08:
            x_dot = 0.10
        #if dist2row[0] < 100:
        #    theta_dot = -0.08
        #    x_dot = 0.08
        #if dist2row[1] < 100:
        #    theta_dot = 0.08
        #    x_dot = 0.08
        print("left row dist =", dist2row[0], "right row dist =", dist2row[1])
        print("left row theta =", dist2row[2], "right row theta =", dist2row[3])
        print("x_dot =", x_dot, "theta_dot =", theta_dot)
        data = [dist2row[0], dist2row[1], dist2row[2], dist2row[3], x_dot, theta_dot]
        csvwrite.writerow(data)
        to_convert = [np.round(x_dot, 3), np.round(theta_dot, 3)]
        pdt = ik.convert(to_convert)
        pdc = k.getPdCurrent()
        pid.driveClosedLoop(pdt, pdc)

    except KeyboardInterrupt:
        m.MotorL(0)
        m.MotorR(0)
        L1tof.cleanup()
        print('Exiting on Keyboard Interrupt.')
        break
