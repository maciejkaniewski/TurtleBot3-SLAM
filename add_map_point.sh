#!/bin/bash

# Check if ROS2 environment is sourced
if [ -z "$ROS_DISTRO" ]; then
  echo "ROS2 environment is not sourced. Please source your ROS2 setup file (e.g., source /opt/ros/<distro>/setup.bash)"
  exit 1
fi

echo "Adding a map point to the map..."

# Execute the first ROS2 service call
echo "Calling /trigger_pf service..."
ros2 service call /trigger_pf example_interfaces/srv/Trigger "{}"
if [ $? -ne 0 ]; then
  echo "Failed to call /triggerp service."
  exit 1
fi

# Execute the second ROS2 service call
echo "Calling /trigger_hf service..."
ros2 service call /trigger_hf example_interfaces/srv/Trigger "{}"
if [ $? -ne 0 ]; then
  echo "Failed to call /trigger service."
  exit 1
fi

echo "Both services have been called successfully."

