# Mechaholics
Mechaholics Capstone
This Repo contains the codebase for the Mechaholics Capstone project, the AMAR Robot, at Texas A&M University
Data for image classification model building can be found on the team's google drive.


STEPS BEFORE RUN:
http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv
For each message type in the github source code, we must add these message types to Catkin via the tutorial above

#In  CMakeLists.txt, add a function under catkin_package for each python script we will execute(space delineated), for example:

catkin_install_python(PROGRAMS scripts/publisher_node.py scripts/subscriber_node.py
    DESTINATION $(CATKIN_PACKAGE_BIN_DESTINATION)
)