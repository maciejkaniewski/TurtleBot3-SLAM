import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config_map_scanner = os.path.join(
        get_package_share_directory("utils"), "config", "map_scanner_params.yaml"
    )

    map_scanner_node = Node(
        package="utils",
        executable="map_scanner",
        name="map_scanner",
        parameters=[config_map_scanner],
        output="screen",
    )

    ld = LaunchDescription()
    ld.add_action(map_scanner_node)

    return ld
