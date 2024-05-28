#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from std_srvs.srv import Empty


class SrvHandlerPhysics:
    def __init__(self, node: Node):
        """
        Initializes an instance of the SrvHandlerPhysics class.

        Args:
            node (Node): The ROS2 node object.
        """

        self.node = node
        self.pause_physics_client = node.create_client(Empty, "/pause_physics")
        self.unpause_physics_client = node.create_client(Empty, "/unpause_physics")

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

    def pause_physics(self) -> None:
        """
        Pauses the physics.
        """

        self.wait_for_service(self.pause_physics_client, "/pause_physics")
        request = Empty.Request()
        future = self.pause_physics_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        self.handle_service_result(future, "/pasue_physics")

    def unpause_physics(self):
        """
        Unpauses the physics.
        """

        self.wait_for_service(self.unpause_physics_client, "/unpause_physics")
        request = Empty.Request()
        future = self.unpause_physics_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        self.handle_service_result(future, "/unpause_physics")
