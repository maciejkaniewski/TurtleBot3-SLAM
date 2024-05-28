# TurtleBot3-SLAM

TurtleBot3 SLAM is a project I realized in the third semester of my studies as a master thesis at the Wroclaw University of Technology in the field of Control Engineering and Robotics.

## Setup


Please make sure to install `ROS2 Humble Hawksbill` according to the instructions provided on the official website at [Installation - ROS2 Documentation: Humble](https://docs.ros.org/en/humble/Installation.html). Additionaly please install `Gazebo` simulator and related packages. The version of Gazebo used in this project is:
    
```bash
Gazebo multi-robot simulator, version 11.10.2
Copyright (C) 2012 Open Source Robotics Foundation.
Released under the Apache 2 License.
http://gazebosim.org
```

Clone the repository:

```bash
git clone git@github.com:maciejkaniewski/TurtleBot3-SLAM.git
```

Init and update the submodules:

```bash
cd TurtleBot3-SLAM  
```
    
```bash
git submodule update --init --recursive
```

Build the workspace:

```bash
cd ros2_ws
```

```bash
colcon build --symlink-install
```

Source the workspace:

```bash
source install/setup.bash
```

To verify whether the repository has been correctly configured, run the simulation by executing the following command:

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Run teleoperation node:

```bash
ros2 run turtlebot3_teleop teleop_keyboard
```