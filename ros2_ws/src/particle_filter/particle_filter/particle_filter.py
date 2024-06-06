#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class ParticleFilter(Node):
    """
    Class representing the particle filter node.
    """

    def __init__(self):
        """
        Initializes an instance of the ParticleFilter class.
        """

        super().__init__("particle_filter")
        self.get_logger().info("particle_filter node started.")


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()


if __name__ == "__main__":
    main()
