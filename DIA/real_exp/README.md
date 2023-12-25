<h1>Experiment on Real Robot</h1>`

# Table of Contents
- 1 [Real Robot Exp](#Experiment on Real Robot)
    - 1.1 [Setup](#Setup)
    - 1.2 [Run Exp](#Run Experiments)
----

# Setup
* [Install ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) (Unbuntu 20.04)


* [Install Universal_Robots_ROS_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)

```
# clone the driver
$ git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git src/Universal_Robots_ROS_Driver

# clone the description. Currently, it is necessary to use the melodic-devel branch.
$ git clone -b melodic-devel https://github.com/ros-industrial/universal_robot.git src/universal_robot

# install dependencies
$ sudo apt update -qq
$ rosdep update
$ rosdep install --from-paths src --ignore-src -y

# build the workspace
$ catkin_make

# activate the workspace (ie: source it)
$ source devel/setup.bash
```
* [Install Realsense ROS Driver](https://github.com/IntelRealSense/realsense-ros/blob/ros1-legacy/README.md#installation-instructions)
      
  - Step 1: Install the latest Intel&reg; RealSense&trade; SDK 2.0
     -  Build from sources by downloading the latest [Intel&reg; RealSense&trade; SDK 2.0](https://github.com/IntelRealSense/librealsense/releases/tag/v2.50.0) and follow the instructions under [Linux Installation](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)
  - Step 2: Install Intel&reg; RealSense&trade; ROS from Sources

```
$ git clone https://github.com/IntelRealSense/realsense-ros.git
$ cd realsense-ros/
$ git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1`
$ cd ..

$ catkin_init_workspace
$ cd ..
$ catkin_make clean
$ catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
$ catkin_make install
```

# Run Experiments

## To start the camera node in ROS:

```
roslaunch realsense2_camera rs_camera.launch filters:=pointcloud
```
