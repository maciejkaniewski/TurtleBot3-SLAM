#!/usr/bin/env python3


import numpy as np
import rclpy
import tf_transformations
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from particle_filter.particle import Particle
#from particle import Particle # for debugging
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
        self.declare_parameter("num_particles", 10)
        self.declare_parameter("x_range_m", [-2.25, 2.25])
        self.declare_parameter("y_range_m", [-2.25, 2.25])
        self.declare_parameter("theta_range_deg", [0, 0])
        self.declare_parameter("odom_topic", "/odom_vel")
        self.declare_parameter("odom_std", [0.0, 0.0])

        # Get parameters
        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value
        self.x_range_m = tuple(self.get_parameter("x_range_m").get_parameter_value().double_array_value)
        self.y_range_m = tuple(self.get_parameter("y_range_m").get_parameter_value().double_array_value)
        self.theta_range_deg = tuple(self.get_parameter("theta_range_deg").get_parameter_value().integer_array_value)
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.odom_std = tuple(self.get_parameter("odom_std").get_parameter_value().double_array_value)

        # Log the parameters
        self.get_logger().info(f"num_particles: {self.num_particles}")
        self.get_logger().info(f"x_range_m: {self.x_range_m}")
        self.get_logger().info(f"y_range_m: {self.y_range_m}")
        self.get_logger().info(f"theta_range_deg: {self.theta_range_deg}")
        self.get_logger().info(f"odom_topic: {self.odom_topic}")
        self.get_logger().info(f"odom_std: {self.odom_std}")

        # Create /parameter_events and selected odom topic subscription
        self.event_subscription = self.create_subscription(ParameterEvent, '/parameter_events', self.parameter_event_callback, 10)
        self.odom_subscription = self.create_subscription(PoseStamped, self.odom_topic, self.odom_callback, 10)

        # Create /particles_poses publisher with a 30 Hz timer
        self.particles_poses_publisher = self.create_publisher(PoseArray, "/particles_poses", 10)
        self.timer_particles_poses = self.create_timer(1 / 30, self.particles_poses_callback)

        # Create particles
        self.particles = self.initialize_particles(self.num_particles, self.x_range_m, self.y_range_m, self.theta_range_deg)
        self.particles_poses = self.particles_to_poses(self.particles)

        # Initialize the previous odometry pose
        self.previous_odom_pose = PoseStamped()
        self.previous_odom_pose_initialized = False

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
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            theta = np.random.uniform(np.deg2rad(theta_range[0]), np.deg2rad(theta_range[1])) % (2 * np.pi)
            particles.append(Particle(x, y, theta, 1.0 / num_particles))
        return particles
    
    def predict_particles(self, particles: list, pose_msg: PoseStamped, std=(0.0, 0.0)) -> list:
        """
        Predicts the particles based on the odometry data.

        Args:
            particles (list): The list of particles.
            pose_msg (PoseStamped): The odometry data.
            std (tuple, optional): The standard deviation of the odometry data. 
        """

        pose_std, angle_std = std

        delta_x = abs(pose_msg.pose.position.x - self.previous_odom_pose.pose.position.x)
        delta_y = abs(pose_msg.pose.position.y - self.previous_odom_pose.pose.position.y)

        _, _, previous_yaw = tf_transformations.euler_from_quaternion([
            self.previous_odom_pose.pose.orientation.x,
            self.previous_odom_pose.pose.orientation.y,
            self.previous_odom_pose.pose.orientation.z,
            self.previous_odom_pose.pose.orientation.w
        ])

        _, _, current_yaw= tf_transformations.euler_from_quaternion([
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        ])

        delta_theta = current_yaw - previous_yaw

        for particle in particles:
            particle.x += delta_x * np.cos(particle.theta) + np.random.normal(0, pose_std)
            particle.y += delta_y * np.sin(particle.theta) + np.random.normal(0, pose_std)
            particle.theta += delta_theta + np.random.normal(0, angle_std)
            particle.theta = np.arctan2(np.sin(particle.theta), np.cos(particle.theta))

        self.previous_odom_pose = pose_msg

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

    def odom_callback(self, msg: PoseStamped) -> None:
        """
        Callback function for the odometry velocity publisher.
        """

        if self.previous_odom_pose_initialized is False:
            self.previous_odom_pose = msg
            self.previous_odom_pose_initialized = True

        self.predict_particles(self.particles, msg, self.odom_std)
    
    def parameter_event_callback(self, event: ParameterEvent) -> None:
        """
        Callback function for handling parameter events.
        """

        for changed_parameter in event.changed_parameters:
            if changed_parameter.name == "num_particles":
                self.get_logger().info(f"num_particles changed to: {changed_parameter.value.integer_value}")
                self.num_particles = changed_parameter.value.integer_value
                self.particles = self.initialize_particles(self.num_particles, self.x_range_m, self.y_range_m, self.theta_range_deg)
            elif changed_parameter.name == "x_range_m":
                self.get_logger().info(f"x_range_m changed to: {changed_parameter.value.double_array_value}")
                self.x_range_m = tuple(changed_parameter.value.double_array_value)
                self.particles = self.initialize_particles(self.num_particles, self.x_range_m, self.y_range_m, self.theta_range_deg)
            elif changed_parameter.name == "y_range_m":
                self.get_logger().info(f"y_range_m changed to: {changed_parameter.value.double_array_value}")
                self.y_range_m = tuple(changed_parameter.value.double_array_value)
                self.particles = self.initialize_particles(self.num_particles, self.x_range_m, self.y_range_m, self.theta_range_deg)
            elif changed_parameter.name == "theta_range_deg":
                self.get_logger().info(f"theta_range_deg changed to: {changed_parameter.value.integer_array_value}")
                self.theta_range_deg = tuple(changed_parameter.value.integer_array_value)
                self.particles = self.initialize_particles(self.num_particles, self.x_range_m, self.y_range_m, self.theta_range_deg)
            elif changed_parameter.name == "odom_topic":
                self.get_logger().info(f"odom_topic changed to: {changed_parameter.value.string_value}")
                self.odom_topic = changed_parameter.value.string_value
                self.destroy_subscription(self.odom_subscription)
                self.odom_subscription = self.create_subscription(PoseStamped, self.odom_topic, self.odom_callback, 10)
            elif changed_parameter.name == "odom_std":
                self.get_logger().info(f"odom_std changed to: {changed_parameter.value.double_array_value}")
                self.odom_std = tuple(changed_parameter.value.double_array_value)

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()


if __name__ == "__main__":
    main()
