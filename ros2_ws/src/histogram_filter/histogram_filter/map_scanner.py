#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class MapScanner(Node):
    def __init__(self):
        super().__init__("map_scanner")
        self.get_logger().info("MapScanner node started.")


def main(args=None):
    rclpy.init(args=args)
    node = MapScanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
