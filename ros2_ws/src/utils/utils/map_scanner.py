#!/usr/bin/env python3

import os
import pickle
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from utils.map_loader import MapLoader
from utils.scan_data import ScanData
from utils.srv_handler_entity import SrvHandlerEntity
from utils.srv_handler_physics import SrvHandlerPhysics


class MapScanner(Node):
    """
    Class representing a map scanner node.

    This node is responsible for scanning the map with given resolution and range.
    """

    def __init__(self):
        """
        Initializes an instance of the MapScanner class.
        """

        super().__init__("map_scanner")
        self.get_logger().info("MapScanner node started.")

        # Declare parameters with default values
        self.declare_parameter('robot_width_m', 0.178)
        self.declare_parameter('robot_length_m', 0.14)
        self.declare_parameter('robot_center_offset_m', 0.032)
        self.declare_parameter('map_pgm_file', 'turtlebot3_dqn_stage4.pgm')
        self.declare_parameter('map_yaml_file', 'turtlebot3_dqn_stage4.yaml')
        self.declare_parameter('scan_data_name', 'scan_data.pkl')
        self.declare_parameter('scan_step_m', 0.1)

        # Get parameters
        self.robot_width_m = self.get_parameter('robot_width_m').get_parameter_value().double_value
        self.robot_length_m = self.get_parameter('robot_length_m').get_parameter_value().double_value
        self.robot_center_offset_m = self.get_parameter('robot_center_offset_m').get_parameter_value().double_value
        self.map_pgm_file = self.get_parameter('map_pgm_file').get_parameter_value().string_value
        self.map_yaml_file = self.get_parameter('map_yaml_file').get_parameter_value().string_value
        self.scan_data_name = self.get_parameter('scan_data_name').get_parameter_value().string_value
        self.scan_step_m = self.get_parameter('scan_step_m').get_parameter_value().double_value

        # Log the parameters
        self.get_logger().info(f"robot_width_m: {self.robot_width_m}")
        self.get_logger().info(f"robot_length_m: {self.robot_length_m}")
        self.get_logger().info(f"robot_center_offset_m: {self.robot_center_offset_m}")
        self.get_logger().info(f"map_pgm_file: {self.map_pgm_file}")
        self.get_logger().info(f"map_yaml_file: {self.map_yaml_file}")
        self.get_logger().info(f"scan_data_name: {self.scan_data_name}")
        self.get_logger().info(f"scan_step_m: {self.scan_step_m}")

        # Get the directory of the package source directory
        package_source_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.scan_data_path = os.path.join(package_source_directory, "maps_data", self.scan_data_name)

        # Create service handlers
        self.entity_srv_handler = SrvHandlerEntity(self)
        self.physics_srv_handler = SrvHandlerPhysics(self)

        # Create /scan subscription
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # Robot position
        self.robot_x_m = 0.0
        self.robot_y_m = 0.0
        self.robot_theta_deg = 0.0
        
        # Scan data
        self.scan_data = np.array([])
        self.new_scan_received = True
        self.scan_counter = 0

    def load_map(self) -> None:
        """
        Loads the map from the specified PGM and YAML files.
        """

        world_pgm_path = os.path.join(
            get_package_share_directory("utils"),
            "maps_pgm",
            self.map_pgm_file
        )

        world_yaml_path = os.path.join(
            get_package_share_directory("utils"),
            "maps_yaml",
            self.map_yaml_file
        )

        pgm_map = MapLoader(world_yaml_path, world_pgm_path)
        self.obstacle_coordinates = pgm_map.get_obstacle_coordinates()
        self.obstacle_radius = pgm_map.get_obstacle_radius()

    def save_scan_data(self, file_path: str) -> None:
        """
        Save the scan data to a file using pickle.

        Args:
            file_path (str): The file path to the output file.
        """

        with open(file_path, "wb") as file:
            pickle.dump(self.scan_data, file)

    def set_entity_state(self, x_set: float, y_set: float, theta_set: float) -> None:
        """
        Sets the entity state to the specified position and orientation.

        Args:
            x_set (float): The x-coordinate of the position.
            y_set (float): The y-coordinate of the position.
            theta_set (float): The orientation angle in degrees.
        """

        self.robot_x_m = x_set
        self.robot_y_m = y_set
        self.robot_theta_deg = theta_set
        self.entity_srv_handler.send_request(x_set, y_set, theta_set)

    def pause_physics(self) -> None:
        """
        Pauses the physics using the physics service handler.
        """

        self.physics_srv_handler.pause_physics()

    def unpause_physics(self) -> None:
        """
        Unpauses the physics using the physics service handler.
        """

        self.physics_srv_handler.unpause_physics()
    
    def is_position_near_obstacle(self, x: float, y: float) -> bool:
        """
        Checks if the given position is near any obstacle, considering the robot's size.

        Args:
            x (float): The x-coordinate of the robot's center.
            y (float): The y-coordinate of the robot's center.

        Returns:
            bool: True if the position is near an obstacle, False otherwise.
        """

        # fmt: off
        robot_half_width = self.robot_width_m / 2
        robot_top_length = (self.robot_length_m / 2 - self.robot_center_offset_m)
        robot_bottom_length = (self.robot_length_m / 2 + self.robot_center_offset_m)

        for obstacle_x, obstacle_y in self.obstacle_coordinates:
            y_distance = abs(y - obstacle_y)
            threshold = (robot_top_length + self.obstacle_radius if y < obstacle_y else robot_bottom_length - self.obstacle_radius)
            if (y_distance < threshold and abs(x - obstacle_x) < robot_half_width + self.obstacle_radius):
                return True

        return False

    def spawn_robot_across_map(self, step: float, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """
        Spawns the robot across the map with a specified step, avoiding obstacles.

        Args:
            step (float): The step size in meters for iterating across the map.
            x_min (float): The minimum x-coordinate for spawning the robot.
            x_max (float): The maximum x-coordinate for spawning the robot.
            y_min (float): The minimum y-coordinate for spawning the robot.
            y_max (float): The maximum y-coordinate for spawning the robot.
        """
        
        for y in np.arange(y_min, y_max + step, step):
            for x in np.arange(x_min, x_max + step, step):
                if not self.is_position_near_obstacle(x, y):
                    #self.pause_physics()
                    self.set_entity_state(x, y, 90)
                    #self.unpause_physics()
                    time.sleep(1)
                    self.new_scan_received = False
                    while not self.new_scan_received:
                        rclpy.spin_once(self)

        self.get_logger().info(f"Saving scan data at {self.scan_data_path}")
        self.save_scan_data(self.scan_data_path)

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback function for the laser scan message.

        Args:
            msg (LaserScan): The laser scan message.
        """
        
        if not self.new_scan_received:
            scan_data = ScanData(
                position=(self.robot_x_m, self.robot_y_m, self.robot_theta_deg),
                measurements=msg.ranges,
            )
            self.scan_data = np.append(self.scan_data, scan_data)
            self.new_scan_received = True

            self.scan_counter += 1
            self.get_logger().info(f"[{self.scan_counter}] scan taken at position: (X: {scan_data.position[0]:.2f} [m], Y: {scan_data.position[1]:.2f} [m], θ: {scan_data.position[2]:.2f} [°])")


def main(args=None):
    rclpy.init(args=args)
    node = MapScanner()
    node.load_map()
    try:
        node.spawn_robot_across_map(step=node.scan_step_m, x_min=-3, x_max=3, y_min=-3, y_max=3)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (Ctrl+C) detected. Shutting down...")
        node.get_logger().info(f"Saving scan data at {node.scan_data_path}")
        node.save_scan_data(node.scan_data_path)
        node.destroy_node()

if __name__ == "__main__":
    main()
