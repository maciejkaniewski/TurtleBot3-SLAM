#!/usr/bin/env python3

import numpy as np
import rclpy
import tf_transformations
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState


class OdomVelocity(Node):
    """
    Class representing the odometry velocity node.

    This node calculates the odometry of a robot based on the wheel velocities
    and publishes the odometry information to the `/odom_vel` topic.
    """

    def __init__(self):
        """
        Initializes an instance of the OdomVelocity class.
        """

        super().__init__("odom_vel")
        self.get_logger().info("odom_vel node started.")

        # Declare parameters with default values
        self.declare_parameter('left_wheel_indx', 0)
        self.declare_parameter('right_wheel_indx', 1)
        self.declare_parameter('wheel_radius_m', 0.033)
        self.declare_parameter('distance_between_wheels_m', 0.160)

        # Get parameters
        self.left_wheel_indx = self.get_parameter('left_wheel_indx').get_parameter_value().integer_value
        self.right_wheel_indx= self.get_parameter('right_wheel_indx').get_parameter_value().integer_value
        self.wheel_radius_m = self.get_parameter('wheel_radius_m').get_parameter_value().double_value
        self.distance_between_wheels_m = self.get_parameter('distance_between_wheels_m').get_parameter_value().double_value

        # Log the parameters
        self.get_logger().info(f"left_wheel_indx: {self.left_wheel_indx}")
        self.get_logger().info(f"right_wheel_indx: {self.right_wheel_indx}")
        self.get_logger().info(f"wheel_radius_m: {self.wheel_radius_m}")
        self.get_logger().info(f"distance_between_wheels_m: {self.distance_between_wheels_m}")

        # Create /odom and /joint_states subscriptions
        self.odom_subscription = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.joint_states_subscription = self.create_subscription(JointState, "/joint_states", self.joint_states_callback, 10)
        
        # Create /odom_vel publisher with a 30 Hz timer
        self.odom_vel_publisher = self.create_publisher(PoseStamped, "/odom_vel", 10)
        self.timer_odom_vel = self.create_timer(1 / 30, self.odom_vel_callback)

        # Robot position
        self.robot_x_m = 0.0
        self.robot_y_m = 0.0
        self.robot_theta_rad = 0.0
        self.robot_theta_deg = 0.0

        # Wheel velocities
        self.right_wheel_velocity_m_s = 0.0
        self.left_wheel_velocity_m_s = 0.0
        self.right_wheel_velocity_rad_s = 0.0
        self.left_wheel_velocity_rad_s = 0.0

        # Previous time for odometry calculation
        self.prev_time_ros = None

    def calculate_odometry_from_velocities(self, current_time_ros: Time) -> None:
        """
        Calculates the odometry of the robot based on the wheel velocities.

        Args:
            current_time_ros (Time): The current time in ROS.
        """

        if self.prev_time_ros is not None:
            
            d_time = (current_time_ros - self.prev_time_ros).nanoseconds / 1e9

            # fmt: off
            d_theta = ((self.right_wheel_velocity_m_s - self.left_wheel_velocity_m_s) / self.distance_between_wheels_m * d_time)
            d_x = ((self.left_wheel_velocity_m_s + self.right_wheel_velocity_m_s) / 2 * d_time * np.cos(self.robot_theta_rad))
            d_y = ((self.left_wheel_velocity_m_s + self.right_wheel_velocity_m_s) / 2 * d_time * np.sin(self.robot_theta_rad))

            self.robot_x_m += d_x
            self.robot_y_m += d_y
            self.robot_theta_rad += d_theta

            # Normalize the angle to the range [-pi, pi]
            self.robot_theta_rad = np.arctan2(np.sin(self.robot_theta_rad), np.cos(self.robot_theta_rad))
            self.robot_theta_deg = np.degrees(self.robot_theta_rad)
            # fmt: on

        self.prev_time_ros = current_time_ros

    def odom_vel_callback(self) -> None:
        """
        Callback function for the odometry velocity publisher.
        """

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "odom"

        pose = pose_stamped.pose
        pose.position.x = self.robot_x_m
        pose.position.y = self.robot_y_m

        # fmt: off
        quaternion = tf_transformations.quaternion_from_euler(0, 0, self.robot_theta_rad)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        # fmt: on
        self.odom_vel_publisher.publish(pose_stamped)

    def joint_states_callback(self, msg: JointState) -> None:
        """
        Callback function for the joint states message.

        Args:
            msg (JointState): The joint states message.
        """

        self.left_wheel_velocity_rad_s = msg.velocity[self.left_wheel_indx]
        self.right_wheel_velocity_rad_s = msg.velocity[self.right_wheel_indx]

        # Convert the velocities to m/s
        self.left_wheel_velocity_m_s = self.left_wheel_velocity_rad_s * self.wheel_radius_m
        self.right_wheel_velocity_m_s = self.right_wheel_velocity_rad_s * self.wheel_radius_m

        current_time_ros = Time.from_msg(msg.header.stamp)
        self.calculate_odometry_from_velocities(current_time_ros)

        # Log the position
        # self.get_logger().info(f"Robot position: (X: {self.robot_x_m:.2f} [m], Y: {self.robot_y_m:.2f} [m], θ: {np.degrees(self.robot_theta_rad):.2f} [°])")

    def odom_callback(self, msg: Odometry) -> None:
        """
        Callback function for handling odometry messages.

        Args:
            msg (Odometry): The incoming odometry message.
        """

        self.robot_x_m = msg.pose.pose.position.x
        self.robot_y_m = msg.pose.pose.position.y

        # fmt: off
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w,]
        _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
        self.robot_theta_rad = yaw
        # fmt: on

        self.destroy_subscription(self.odom_subscription)
        self.get_logger().info("Unsubscribed from /odom topic.")
        self.get_logger().info("Publishing to /odom_vel topic.")


def main(args=None):
    rclpy.init(args=args)
    node = OdomVelocity()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()


if __name__ == "__main__":
    main()
