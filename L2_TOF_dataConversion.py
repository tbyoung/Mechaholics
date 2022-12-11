import L1_TOF as TOF
import numpy as np
import os
import time as t

# min_dist = # enter minimum TOF 

def dist2rowapprox():
    # takes input from the 4 time-of-flight sensors. returns the approx. distance to each row, as well as
    # the angle of each row relative to the current orientation.
    distance = TOF.getRange()

    tof_spacing = 190 # Enter the space between the laser on each ToF sensor here in mm.
    tof_angle = np.radians(0) # Enter the rotational angle of the 4 ToF sensors here. Try to keep all angles identical.
                              # Perpendicular to driving direction = 0 deg.
    out = []
    for i in range(len(distance)):
        if distance[i] == -1:
            TOF.reset(i)
    FL = distance[3]
    FR = distance[1]
    BL = distance[0]
    BR = distance[2]
    
    left_theta = np.round(np.arctan(np.cos(tof_angle) * (FL - BL) / (tof_spacing + np.sin(tof_angle) * BL * FL)), 2)

    if left_theta >= 0: # provides shortest distance to crop row based on left TOF angle
        left_dist = np.cos(left_theta) * BL
    else:
        left_dist = np.cos(abs(left_theta)) * FL

    right_theta = np.round(np.arctan(np.cos(tof_angle) * (FR - BR) / (tof_spacing + np.sin(tof_angle) * BR * FR)), 2)
    if right_theta >= 0: # provides shortest distance to crop row based on right TOF angle
        right_dist = np.cos(right_theta) * BR 
    else:
        right_dist = np.cos(abs(left_theta)) * FR

    out.append(left_dist)
    out.append(right_dist)
    out.append(np.degrees(left_theta))
    out.append(np.degrees(right_theta))
    # Positive value for theta = outward angle.
    # robot --> |_|   / <--crop row ->> positive right_theta
    # robot --> |_|   \ <--crop row ->> negative right_theta
    # Theta is in RADIANS!
    return out # return left_dist, right_dist, left_theta, right_theta


global angDist_array
angDist_array = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], 
                          [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], 
                          [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], 
                          [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
def angDist_avg(iter):
    angDist_iterator = iter
    curr = dist2rowapprox()
    if curr[0] > 1000:
        curr[0] = np.nan
        #print("left sensor gap.")
    if curr[1] > 1000:
        curr[1] = np.nan
        #print("right sensor gap.")
    if abs(curr[2] > 45):
        curr[2] = np.nan
        #print('left angle > 60.')
    if abs(curr[3] > 45):
        curr[3] = np.nan
        #print('right angle > 60.')
    angDist_array [0][angDist_iterator] = curr[0]
    angDist_array [1][angDist_iterator] = curr[1]
    angDist_array [2][angDist_iterator] = curr[2]
    angDist_array [3][angDist_iterator] = curr[3]

    left_dist_avg = np.nanmean(angDist_array[0])
    right_dist_avg = np.nanmean(angDist_array[1])
    left_theta_avg = np.nanmean(angDist_array[2])
    right_theta_avg = np.nanmean(angDist_array[3])
    if np.isnan(np.nanmean(angDist_array)):
        # print("Whole array is NAN!")
        left_dist_avg = 50
        right_dist_avg = 50
        left_theta_avg = 0
        right_theta_avg = 0
    if angDist_iterator >= 23:
        angDist_iterator = -1
    avgs = np.array([left_dist_avg, right_dist_avg, left_theta_avg, right_theta_avg])
    angDist_iterator += 1
    return curr, avgs, angDist_iterator
    



# maybe, average the data points over a half second or so before performing these calculations? may help
# smooth these results.

if __name__ == "__main__":
    u_in = int(input('Enter time to run in seconds:'))
    for i in range(u_in * 10):
        start_time = t.time()
        iter = 0
        dnt_need, dist2row, iter = angDist_avg(iter)
        print('----%s seconds to iterate----' % (t.time() - start_time))
        print('Left row distance:', dist2row[0], "mm.")
        print('Right row distance:', dist2row[1], 'mm.')
        print('Left row angle:', dist2row[2], 'degrees.')
        print('Right row angle:', dist2row[3], 'degrees.')
        t.sleep(0.5)
    TOF.cleanup()
    print('Exiting')
