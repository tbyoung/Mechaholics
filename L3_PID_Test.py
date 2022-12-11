import numpy as np
import time
import csv

import L2_Inverse_Kinematics as ik
import L2_Kinematics as k
import L2_PID as pid
import L1_MotorControl as m


if __name__ == "__main__":
    filename = 'PDL.csv'
    filename2 = 'PDR.csv'
    headerL = ['time', 'pdcl', 'pdtl']
    headerR = ['time', 'pdcr', 'pdtr']
    fileL = open(filename, 'w', newline="")
    fileR = open(filename2, 'w', newline = "")
    csvwriterL = csv.writer(fileL)
    csvwriterR = csv.writer(fileR)
    csvwriterL.writerow(headerL)
    csvwriterR.writerow(headerR)
    while True:
        x_dot , theta_dot = ik.wait_user()  # user input [x_dot,theta_dot] (m/s, rad/s)
        mat = [x_dot, theta_dot]
        pdt = ik.convert(mat) # Converts x_dot, theta_dot to  desired PDL, PDR
        print(pdt)
        pdc = k.getPdCurrent() # gets current PDL, PDR
        print(pdc)
        pid.driveClosedLoop(pdt, pdc, 0) # pid controller attempts to match pdl, pdr target     
        timeKeepInitial = time.time()
        #for i in range(200):
        #    pid.driveOpenLoop(pdt)
        for i in range(1000):
            #print("PDC Original: ", pdc)
            # print("Current Motion: ", k.getMotion())
            timeKeep = time.time()-timeKeepInitial
            pdc = k.getPdCurrent()
            pid.driveClosedLoop(pdt, pdc, 0)
            # m.MotorR(5000)
            # m.MotorL(5000)
            dataL = [timeKeep, pdc[0], pdt[0]]
            dataR = [timeKeep, pdc[1], pdt[1]]
            csvwriterL.writerow(dataL)
            csvwriterR.writerow(dataR)
        m.MotorL(0)
        m.MotorR(0)  
