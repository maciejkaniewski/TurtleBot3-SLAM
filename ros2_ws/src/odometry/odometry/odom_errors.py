#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np
import time

class OdomErrors(Node):
    """
    Class representing the odometry errors node.

    This node subscribes to the `/odom_vel`, `/odom_pos`, and `/odom` topics to 
    calculate and plot the errors between the estimated positions and the true position.
    """

    def __init__(self):
        """
        Initializes an instance of the OdomErrors class.
        """
        super().__init__("odom_errors")
        self.get_logger().info("odom_errors node started.")

        # Create subscriptions
        self.odom_vel_subscription = self.create_subscription(PoseStamped, "/odom_vel", self.odom_vel_callback, 10)
        self.odom_pos_subscription = self.create_subscription(PoseStamped, "/odom_pos", self.odom_pos_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # Lists to store positions and their corresponding times
        self.odom_vel_positions = []
        self.odom_pos_positions = []
        self.true_positions = []

        # Start time
        self.start_time = time.time()
        self.end_time = None

    def odom_vel_callback(self, msg: PoseStamped) -> None:
        """
        Callback function for the `/odom_vel` topic.
        
        Args:
            msg (PoseStamped): The odometry velocity message.
        """
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.odom_vel_positions.append((x, y))

    def odom_pos_callback(self, msg: PoseStamped) -> None:
        """
        Callback function for the `/odom_pos` topic.
        
        Args:
            msg (PoseStamped): The odometry position message.
        """
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.odom_pos_positions.append((x, y))

    def odom_callback(self, msg: Odometry) -> None:
        """
        Callback function for the `/odom` topic.
        
        Args:
            msg (Odometry): The true odometry message.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.true_positions.append((x, y))

    def plot_errors(self) -> None:
        """
        Function to plot the errors between estimated positions and the true position.
        """
        if not (self.odom_vel_positions and self.odom_pos_positions and self.true_positions):
            self.get_logger().info("Insufficient data to plot errors.")
            return

        true_x, true_y = zip(*self.true_positions)
        vel_x, vel_y = zip(*self.odom_vel_positions)
        pos_x, pos_y = zip(*self.odom_pos_positions)

        # Calculate errors using numpy.linalg.norm
        vel_errors = [np.linalg.norm([true_x[i] - vel_x[i], true_y[i] - vel_y[i]]) for i in range(len(true_x))]
        pos_errors = [np.linalg.norm([true_x[i] - pos_x[i], true_y[i] - pos_y[i]]) for i in range(len(true_x))]

        # Calculate total duration in seconds
        total_time_seconds = self.end_time - self.start_time if self.end_time else len(vel_errors) / 30  # Fallback to data length / 30 Hz
        num_points = min(len(vel_errors), len(pos_errors))
        time_step = total_time_seconds / num_points

        # Generate x-axis based on total duration
        times = np.linspace(0, total_time_seconds, num_points)

        # Plot errors with time as the x-axis
        plt.figure(figsize=(10, 6))
        plt.plot(times, vel_errors[:num_points], label='Error /odom_vel', color='blue')
        plt.plot(times, pos_errors[:num_points], label='Error /odom_pos', color='red')
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Euclidean Distance Error [m]', fontsize=12)
        plt.title('Odometry Error Analysis over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = OdomErrors()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.end_time = time.time()  # Record end time when interrupted
        node.get_logger().info(f"Total duration: {node.end_time - node.start_time:.2f} [s]")
        node.get_logger().info("Plotting errors...")
        node.plot_errors()
        node.destroy_node()

if __name__ == "__main__":
    main()
