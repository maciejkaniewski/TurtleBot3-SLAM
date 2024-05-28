#!/usr/bin/env python3

import os
import pickle
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from config.robot import *
from utils.map_loader import MapLoader
from utils.scan_data import ScanData
from utils.srv_handler_entity import SrvHandlerEntity
from utils.srv_handler_physics import SrvHandlerPhysics


class MapScanner(Node):
    def __init__(self):
        super().__init__("map_scanner")
        self.get_logger().info("MapScanner node started.")
        self.entity_srv_handler = SrvHandlerEntity(self)
        self.physics_srv_handler = SrvHandlerPhysics(self)

        self.scan_subscription = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.laser_scan_data = np.array([])

    def load_map(self) -> None:
        """
        Loads the map from the specified PGM and YAML files.
        """

        world_pgm_path = os.path.join(
            get_package_share_directory("utils"),
            "maps_pgm",
            "turtlebot3_dqn_stage4.pgm",
        )

        world_yaml_path = os.path.join(
            get_package_share_directory("utils"),
            "maps_yaml",
            "turtlebot3_dqn_stage4.yaml",
        )

        pgm_map = MapLoader(world_yaml_path, world_pgm_path)
        self.obstacle_coordinates = pgm_map.get_obstacle_coordinates()
        self.obstacle_radius = pgm_map.get_obstacle_radius()

    def save_laser_scan_data(self, file_path: str) -> None:
        """
        Save the laser scan data to a file using pickle.

        Args:
            file_path (str): The file path to the output file.
        """

        with open(file_path, "wb") as file:
            pickle.dump(self.laser_scan_data, file)

    def set_entity_state(self, x_set: float, y_set: float, theta_set: float) -> None:

        self.robot_x = x_set
        self.robot_y = y_set
        self.robot_theta = theta_set
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
        robot_half_width = TURTLEBOT3_BURGER_WIDTH_M / 2
        robot_top_length = (TURTLEBOT3_BURGER_LENGTH_M / 2 - TURTLEBOT3_BURGER_CENTER_OFFSET_M)
        robot_bottom_length = (TURTLEBOT3_BURGER_LENGTH_M / 2 + TURTLEBOT3_BURGER_CENTER_OFFSET_M)

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
                    self.pause_physics()
                    self.set_entity_state(x, y, 90)
                    self.unpause_physics()
                    time.sleep(1)
                    rclpy.spin_once(self)

    def scan_callback(self, msg: LaserScan) -> None:

        scan_data = ScanData(
            position=(self.robot_x, self.robot_y, self.robot_theta),
            measurements=msg.ranges,
        )
        self.laser_scan_data = np.append(self.laser_scan_data, scan_data)


def main(args=None):
    rclpy.init(args=args)
    node = MapScanner()
    node.load_map()
    node.spawn_robot_across_map(step=0.1, x_min=-3, x_max=3, y_min=2.1, y_max=3)
    node.save_laser_scan_data("/home/mkaniews/Desktop/test_6.pkl")
    node.destroy_node()


if __name__ == "__main__":
    main()
