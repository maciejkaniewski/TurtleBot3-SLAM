import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config_odom_vel = os.path.join(
        get_package_share_directory("odometry"), "config", "odom_vel_params.yaml"
    )

    config_odom_pos = os.path.join(
        get_package_share_directory("odometry"), "config", "odom_pos_params.yaml"
    )

    rviz_config_dir = os.path.join(
        get_package_share_directory("histogram_filter"), "config", "rviz2_conf.rviz"
    )

    config_histogram_filter = os.path.join(
        get_package_share_directory("histogram_filter"), "config", "histogram_filter_params.yaml"
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_dir],
        output="screen",
    )

    odom_vel_node = Node(
        package="odometry",
        name="odom_vel",
        executable="odom_vel",
        parameters=[config_odom_vel],
        output="screen",
    )

    odom_pos_node = Node(
        package="odometry",
        name="odom_pos",
        executable="odom_pos",
        parameters=[config_odom_pos],
        output="screen",
    )

    histogram_filter_node = Node(
        package="histogram_filter",
        name="histogram_filter",
        executable="histogram_filter",
        parameters=[config_histogram_filter],
        output="screen",
    )

    ld = LaunchDescription()
    ld.add_action(odom_vel_node)
    ld.add_action(odom_pos_node)
    ld.add_action(rviz_node)
    ld.add_action(histogram_filter_node)

    return ld
