import numpy as np
import L1_Lidar as liscan
import math as m
#For testing purposes
import pandas as pd

#Downward angle of the 561 Lidar, from straight ahead, in degrees
ground = 0.25 # y measurement for ground distance from LiDAR

#Number of scan points that you want the LiDAR to make

def pol2cart(dist, ang):
	x = np.cos(np.radians(ang)) * dist
	y = np.sin(np.radians(ang)) * dist
	return(x, y)

def lidar_xy(num_points=54):
	scan = liscan.polarScan(num_points)
	dists, angs = np.hsplit(scan, 2)
	x, y = pol2cart(dists, angs)
	xy = np.hstack((x, y))
	return xy
	# *** +X IS THE DRIVING DIRECTION! X=0 IS DIRECTLY IN FRONT OF LiDAR

	# At this stage, we have the x and y coordinates of the data points in from LiDAR.
	# Next, we need to isolate relevant data (i.e. remove outliers, ground in front)
	# and create 2 linear fit lines for the left and right rows of plants. Finally,
	# we will output the angle and distance to each linear fit line to show the angle
	# and distance from each plant row.

def find_closest_obstacles(num_points=54):
	liscan = lidar_xy(num_points)
	closest_L_obstacle = 100
	closest_R_obstacle = -100
	for i in range(num_points): 
		if liscan[i][1] > 0:
			if liscan[i][0] < ground and liscan[i][0] > 0.1 and liscan[i][1] > 0.05: # filters out weird data points from pysicktim
				if (liscan[i][1] < closest_L_obstacle):
					closest_L_obstacle = liscan[i][1] # distance from robot to closest left obstacle
		if liscan[i][1] < 0:
			if liscan[i][0] < ground and liscan[i][0] > 0.1 and liscan[i][1] < -0.05: # filters out weird data points from pysicktim
				if (liscan[i][1] > closest_R_obstacle):
					closest_R_obstacle = liscan[i][1] # distance from robot to closest right obstacle
	return closest_L_obstacle, closest_R_obstacle
if __name__ == "__main__":
		xy = lidar_xy()
		print(xy)
		np.savetxt("scan.csv", xy, delimiter=',')
