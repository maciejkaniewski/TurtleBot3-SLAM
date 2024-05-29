#!/usr/bin/env python3

import os

import numpy as np
import rclpy
import tf_transformations
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Point, Quaternion
from rclpy.node import Node
from rclpy.task import Future


class SrvHandlerEntity:
    def __init__(self, node: Node):
        """
        Initializes an instance of the SrvHandlerEntity class.

        Args:
            node (Node): The ROS2 node object.
        """
        
        self.node = node
        self.entity_name = os.environ["TURTLEBOT3_MODEL"]
        self.set_entity_client = node.create_client(SetEntityState, "/gazebo/set_entity_state")

    def wait_for_service(self, service_client, service_name):
        """
        Waits for a service to become available.

        Args:
            service_client: The service client object.
            service_name: The name of the service.
        """

        while not service_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(f"Service {service_name} not available, waiting again...")

    def handle_service_result(self, future: Future, service_name: str) -> None:
        """
        Handles the result of a service call.

        Args:
            future (Future): The future object representing the result of the service call.
            service_name (str): The name of the service.
        """

        if future.result() is not None:
            self.node.get_logger().info(f"Service call {service_name} completed successfully!")
        else:
            self.node.get_logger().error(f"Exception while calling service: {future.exception()}")

    def send_request(self, x_set: float, y_set: float, theta_set: float) -> None:
        """
        Sets the entity state.

        Args:
            x_set (float): The x-coordinate.
            y_set (float): The y-coordinate.
            theta_set (float): The orientation angle.
        """

        self.wait_for_service(self.set_entity_client, "/gazebo/set_entity_state")
        request = SetEntityState.Request()
        request.state.name = self.entity_name
        request.state.pose.position = Point(x=x_set, y=y_set, z=0.1)
        quaternion = tf_transformations.quaternion_from_euler(0, 0, np.deg2rad(theta_set))
        request.state.pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])

        future = self.set_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        self.handle_service_result(future, "/gazebo/set_entity_state")