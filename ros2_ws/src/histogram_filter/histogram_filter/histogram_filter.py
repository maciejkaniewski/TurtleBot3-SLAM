#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class HistogramFilter(Node):
    def __init__(self):
        super().__init__("histogram_filter")
        self.get_logger().info("HistogramFilter node started.")


def main(args=None):
    rclpy.init(args=args)
    node = HistogramFilter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
