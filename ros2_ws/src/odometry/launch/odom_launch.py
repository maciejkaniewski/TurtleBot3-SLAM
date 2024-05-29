import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_odom_vel = os.path.join(
        get_package_share_directory("odometry"), "config", "odom_vel_params.yaml"
    )

    return LaunchDescription(
        [
            Node(
                package="odometry",
                name="odom_vel",
                executable="odom_vel",
                parameters=[config_odom_vel],
            )
        ]
    )
