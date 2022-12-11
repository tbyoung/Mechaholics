import L2_TOF_dataConversion as tof
import L2_LiDAR_dataConversion as lidar

global sensor_count
global sensor_data
sensor_count = 0
sensor_data = np.array([[None, None, None, None], [None, None, None, None]])

def obstacleAvoidance():

