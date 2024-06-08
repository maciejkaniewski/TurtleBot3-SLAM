#!/usr/bin/env python3

import numpy as np
import rclpy
import tf_transformations
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState


class OdomPosition(Node):
    """
    Class representing the odometry position node.

    This node calculates the odometry of a robot based on the wheel positions reading
    and publishes the odometry information to the `/odom_pos` topic.
    """

    def __init__(self):
        """
        Initializes an instance of the OdomPosition class.
        """

        super().__init__("odom_pos")
        self.get_logger().info("odom_pose node started.")

        # Declare parameters with default values
        self.declare_parameter('distance_between_wheels_m', 0.160)
        self.declare_parameter('wheel_radius_m', 0.033)
        self.declare_parameter('left_wheel_indx', 0)
        self.declare_parameter('right_wheel_indx', 1)


        # Get parameters
        self.distance_between_wheels_m = self.get_parameter('distance_between_wheels_m').get_parameter_value().double_value
        self.wheel_radius_m = self.get_parameter('wheel_radius_m').get_parameter_value().double_value
        self.left_wheel_indx = self.get_parameter('left_wheel_indx').get_parameter_value().integer_value
        self.right_wheel_indx = self.get_parameter('right_wheel_indx').get_parameter_value().integer_value

        # Log the parameters
        self.get_logger().info(f"distance_between_wheels_m: {self.distance_between_wheels_m}")
        self.get_logger().info(f"wheel_radius_m: {self.wheel_radius_m}")
        self.get_logger().info(f"left_wheel_indx: {self.left_wheel_indx}")
        self.get_logger().info(f"right_wheel_indx: {self.right_wheel_indx}")

        # Create /odom and /joint_states subscriptions
        self.odom_subscription = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.joint_states_subscription = self.create_subscription(JointState, "/joint_states", self.joint_states_callback, 10)
        
        # Create /odom_pos publisher with a 30 Hz timer
        self.odom_pos_publisher = self.create_publisher(PoseStamped, "/odom_pos", 10)
        self.timer_odom_pos = self.create_timer(1 / 30, self.odom_pos_callback)

        # Robot position
        self.robot_x_m = 0.0
        self.robot_y_m = 0.0
        self.robot_theta_rad = 0.0
        self.robot_theta_deg = 0.0

        # Position values
        self.right_wheel_pos_rad = 0.0
        self.left_wheel_pos_rad = 0.0
        self.prev_right_wheel_pos_rad = None
        self.prev_left_wheel_pos_rad = None

    def calculate_odometry_from_positions(self):
        """
        Calculates the odometry of the robot based on the positions.
        """

        if self.prev_left_wheel_pos_rad is not None and self.prev_right_wheel_pos_rad is not None:
            # Calculate wheel displacements
            d_left = self.wheel_radius_m * (self.left_wheel_pos_rad - self.prev_left_wheel_pos_rad)
            d_right = self.wheel_radius_m * (self.right_wheel_pos_rad - self.prev_right_wheel_pos_rad)

            # Calculate robot displacement and change in orientation
            d = (d_left + d_right) / 2
            d_theta = (d_right - d_left) / self.distance_between_wheels_m

            # Update the pose
            self.robot_x_m += d * np.cos(self.robot_theta_rad)
            self.robot_y_m += d * np.sin(self.robot_theta_rad)
            self.robot_theta_rad += d_theta

            # Normalize the angle to the range [-pi, pi]
            self.robot_theta_rad = np.arctan2(np.sin(self.robot_theta_rad), np.cos(self.robot_theta_rad))
            self.robot_theta_deg = np.degrees(self.robot_theta_rad)

    def odom_pos_callback(self) -> None:
        """
        Callback function for the odometry position publisher.
        """

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "odom"

        pose = pose_stamped.pose
        pose.position.x = self.robot_x_m
        pose.position.y = self.robot_y_m

        quaternion = tf_transformations.quaternion_from_euler(0, 0, self.robot_theta_rad)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        self.odom_pos_publisher.publish(pose_stamped)

    def joint_states_callback(self, msg: JointState) -> None:
        """
        Callback function for the joint states message.
        Args:
            msg (JointState): The joint states message.
        """
        
        if self.prev_left_wheel_pos_rad is None or self.prev_right_wheel_pos_rad is None:
            self.prev_right_wheel_pos_rad = msg.position[self.right_wheel_indx]
            self.prev_left_wheel_pos_rad = msg.position[self.left_wheel_indx]
            return

        # Update position values
        self.right_wheel_pos_rad = msg.position[self.right_wheel_indx]
        self.left_wheel_pos_rad = msg.position[self.left_wheel_indx]

        self.calculate_odometry_from_positions()

        # Log robot position
        # self.get_logger().info(f"Robot position: (X: {self.robot_x_m:.2f} [m], Y: {self.robot_y_m:.2f} [m], θ: {np.degrees(self.robot_theta_rad):.2f} [°])")

        # Update previous position values
        self.prev_right_wheel_pos_rad = self.right_wheel_pos_rad
        self.prev_left_wheel_pos_rad = self.left_wheel_pos_rad
    
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
        self.get_logger().info("Publishing to /odom_pos topic.")

def main(args=None):
    rclpy.init(args=args)
    node = OdomPosition()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
