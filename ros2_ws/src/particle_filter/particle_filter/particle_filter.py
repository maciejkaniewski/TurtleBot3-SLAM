#!/usr/bin/env python3

import random

import numpy as np
import rclpy
import tf_transformations
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from particle_filter.particle import Particle
from rcl_interfaces.msg import ParameterEvent
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

        # Declare parameters with default values
        self.declare_parameter("num_particles", 100)
        self.declare_parameter("x_range_m", [-2.25, 2.25])
        self.declare_parameter("y_range_m", [-2.25, 2.25])
        self.declare_parameter("theta_range_deg", [0, 360])

        # Get parameters
        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value
        self.x_range_m = tuple(self.get_parameter("x_range_m").get_parameter_value().double_array_value)
        self.y_range_m = tuple(self.get_parameter("y_range_m").get_parameter_value().double_array_value)
        self.theta_range_deg = tuple(self.get_parameter("theta_range_deg").get_parameter_value().integer_array_value)

        # Log the parameters
        self.get_logger().info(f"num_particles: {self.num_particles}")
        self.get_logger().info(f"x_range_m: {self.x_range_m}")
        self.get_logger().info(f"y_range_m: {self.y_range_m}")
        self.get_logger().info(f"theta_range_deg: {self.theta_range_deg}")

        # Create /parameter_events subscription
        self.event_subscription = self.create_subscription(ParameterEvent, '/parameter_events', self.parameter_event_callback, 10)


        # Create /particles_poses publisher with a 30 Hz timer
        self.particles_poses_publisher = self.create_publisher(PoseArray, "/particles_poses", 10)
        self.timer_particles_poses = self.create_timer(1 / 30, self.particles_poses_callback)

        # Create particles
        self.particles = self.initialize_particles(self.num_particles, self.x_range_m, self.y_range_m, self.theta_range_deg)
        self.particles_poses = self.particles_to_poses(self.particles)

    def initialize_particles(self, num_particles: int, x_range: tuple, y_range: tuple, theta_range: tuple) -> list:
        """
        Initializes the particles with a uniform distribution.

        Args:
            num_particles (int): The number of particles.
            x_range_m (tuple): The x-coordinate range.
            y_range_m (tuple): The y-coordinate range.
            theta_range_deg (tuple): The theta range.

        Returns:
            list: The list of particles.
        """

        particles = []
        for _ in range(num_particles):
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            theta = random.uniform(np.deg2rad(theta_range[0]), np.deg2rad(theta_range[1])) % (2 * np.pi)
            particles.append(Particle(x, y, theta, 1.0 / num_particles))
        return particles
    
    def particles_to_poses(self, particles: list) -> PoseArray:
        """
        Converts the particles to PoseArray.

        Args:
            particles (list): The list of particles.

        Returns:
            PoseArray: The PoseArray of particles.
        """

        poses = PoseArray()
        poses.header.frame_id = "odom"
        for particle in particles:
            pose = Pose()
            pose.position.x = particle.x
            pose.position.y = particle.y
            quaternion = tf_transformations.quaternion_from_euler(0, 0, particle.theta)
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]
            poses.poses.append(pose)
        return poses

    def particles_poses_callback(self) -> None:
        """
        Publishes the particle poses.
        """
        self.particles_poses = self.particles_to_poses(self.particles)
        self.particles_poses_publisher.publish(self.particles_poses)
    
    def parameter_event_callback(self, event: ParameterEvent) -> None:
        """
        Callback function for handling parameter events.
        """

        for changed_parameter in event.changed_parameters:
            if changed_parameter.name == "num_particles":
                self.get_logger().info(f"num_particles changed to: {changed_parameter.value.integer_value}")
                self.num_particles = changed_parameter.value.integer_value
            elif changed_parameter.name == "x_range_m":
                self.get_logger().info(f"x_range_m changed to: {changed_parameter.value.double_array_value}")
                self.x_range_m = tuple(changed_parameter.value.double_array_value)
            elif changed_parameter.name == "y_range_m":
                self.get_logger().info(f"y_range_m changed to: {changed_parameter.value.double_array_value}")
                self.y_range_m = tuple(changed_parameter.value.double_array_value)
            elif changed_parameter.name == "theta_range_deg":
                self.get_logger().info(f"theta_range_deg changed to: {changed_parameter.value.integer_array_value}")
                self.theta_range_deg = tuple(changed_parameter.value.integer_array_value)
        self.particles = self.initialize_particles(self.num_particles, self.x_range_m, self.y_range_m, self.theta_range_deg)

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()


if __name__ == "__main__":
    main()
