#!/usr/bin/env python3

import os
import pickle

import numpy as np
import rclpy
import tf_transformations
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterEvent
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from utils.scan_data import ScanData


class HistogramFilter(Node):
    """
    Class representing the histogram filter node.
    """

    def __init__(self):
        """
        Initializes an instance of the HistogramFilter class.
        """

        super().__init__("histogram_filter")
        self.get_logger().info("histogram_filter node started.")

        # Declare parameters with default values
        self.declare_parameter('histogram_bins', 15)
        self.declare_parameter('histogram_range_m', [0.12, 3.5])
        self.declare_parameter('histogram_comparison', 'euclidean')
        self.declare_parameter('map_pkl_file', 'turtlebot3_dqn_stage4_0.1.pkl')

        # Get parameters
        self.histogram_bins = self.get_parameter('histogram_bins').get_parameter_value().integer_value
        self.histogram_range_m = self.get_parameter('histogram_range_m').get_parameter_value().double_array_value
        self.histogram_comparison = self.get_parameter('histogram_comparison').get_parameter_value().string_value
        self.map_pkl_file = self.get_parameter('map_pkl_file').get_parameter_value().string_value

        self.histogram_range_m = tuple(self.histogram_range_m)

        # Log the parameters
        self.get_logger().info(f"histogram_bins: {self.histogram_bins}")
        self.get_logger().info(f"histogram_range_m: {self.histogram_range_m}")
        self.get_logger().info(f"histogram_comparison: {self.histogram_comparison}")
        self.get_logger().info(f"map_pkl_file: {self.map_pkl_file}")

        # Create /scan and /parameter_events subscriptions
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.event_subscription = self.create_subscription(ParameterEvent, '/parameter_events', self.parameter_event_callback, 10)

        # Create /histogram_pose publisher with a 5 Hz timer
        self.hfliter_pose_publisher = self.create_publisher(PoseStamped, "/hfilter_pose", 10)
        self.hfilter_pose = self.create_timer(1 / 5, self.hfilter_pose_callback)

        # Scan data
        self.scan_data = None
        self.closest_scan_data = None
        self.scan_data_histograms = np.array([])

        # Robot estimated position
        self.robot_x_m = 0.0
        self.robot_y_m = 0.0
        self.robot_theta_rad = 0.0
        self.robot_theta_deg = 0.0

    def load_scan_data(self):
        """
        Loads scan data from a pickle file.
        """

        scan_data_path = os.path.join(
            get_package_share_directory("utils"),
            "maps_data",
            self.map_pkl_file,
        )

        with open(scan_data_path, "rb") as file:
            self.scan_data = pickle.load(file)

    def convert_scan_data_to_histograms(self):
        """
        Converts the loaded scan data to histograms.
        """

        self.scan_data_histograms = np.array([])

        # fmt: off
        for scan_data in self.scan_data:
            hist, _ = np.histogram(scan_data.measurements, range=self.histogram_range_m, bins=self.histogram_bins)
            new_scan_data = ScanData(position=scan_data.position, measurements=hist)
            self.scan_data_histograms = np.append(self.scan_data_histograms, new_scan_data)
        # fmt: on

    def compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compares two histograms using the specified method.
        """

        method = self.histogram_comparison
        if method == 'euclidean':
            return np.sum(np.abs(hist1 - hist2))
        elif method == 'chi_square':
            return np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))
        elif method == 'correlation':
            return -np.corrcoef(hist1, hist2)[0, 1]
        
    def localize_robot(self, current_histogram: np.ndarray) -> tuple[float, float, int]:
        """
        Localizes the robot based on the current histogram.
        """

        min_difference = float("inf")
        estimated_x, estimated_y, best_fit_index = None, None, None

        for i, scan_data in enumerate(self.scan_data_histograms):
            total_difference = self.compare_histograms(scan_data.measurements, current_histogram)

            if total_difference < min_difference:
                min_difference = total_difference
                estimated_x, estimated_y, _ = scan_data.position
                best_fit_index = i

        self.closest_scan_data = self.scan_data[best_fit_index]
        return estimated_x, estimated_y

    def calculate_orientation(self, current_scan_data):
        """
        Calculates the orientation of the current scan data relative to the closest scan data.
        """

        ref_data = self.closest_scan_data.measurements

        # Adjust reference data to 0 degrees orientation by shifting the data
        adjusted_ref_data = np.roll(ref_data, -90)

        # Replace 'inf' values with the maximum valid value from the reference data
        max_valid_value = np.nanmax(ref_data[~np.isinf(ref_data)])  # Maximum value among non-infinite values.
        adjusted_ref_data = np.where(np.isinf(adjusted_ref_data), max_valid_value, adjusted_ref_data)
        current_scan_data_replaced_inf = np.where(np.isinf(current_scan_data), max_valid_value, current_scan_data)

        # Vectorize the computation of the sum of squared differences for all shifts
        diffs = np.array([
            np.sum((adjusted_ref_data - np.roll(current_scan_data_replaced_inf, shift)) ** 2)
            for shift in range(360)
        ])

        best_shift = np.argmin(diffs)
        return -(180 - best_shift)

    def hfilter_pose_callback(self) -> None:
        """
        Callback function for the histogram filter pose publisher.
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

        self.hfliter_pose_publisher.publish(pose_stamped)

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback function for handling laser scan messages.

        Args:
            msg (LaserScan): The incoming laser scan message.
        """
    
        hist, _ = np.histogram(msg.ranges, range=self.histogram_range_m, bins=self.histogram_bins)
        self.robot_x_m, self.robot_y_m = self.localize_robot(hist)
        self.robot_theta_deg = self.calculate_orientation(msg.ranges)
        self.robot_theta_rad = np.radians(self.robot_theta_deg)
        #self.get_logger().info(f"Robot position: (X: {self.robot_x_m:.2f} [m], Y: {self.robot_y_m:.2f} [m], θ: {self.robot_theta_deg:.0f} [°])")
        
    def parameter_event_callback(self, event: ParameterEvent) -> None:
        """
        Callback function for handling parameter events.
        """

        for changed_parameter in event.changed_parameters:
            if changed_parameter.name == 'histogram_bins':
                self.get_logger().info(f"histogram_bins changed to: {changed_parameter.value.integer_value}")
                self.histogram_bins = changed_parameter.value.integer_value
                self.convert_scan_data_to_histograms()
            elif changed_parameter.name == 'histogram_range_m':
                self.get_logger().info(f"histogram_range_m changed to: {changed_parameter.value.double_array_value}")
                self.histogram_range_m = tuple(changed_parameter.value.double_array_value)
                self.convert_scan_data_to_histograms()
            elif changed_parameter.name == 'histogram_comparison':
                self.get_logger().info(f"histogram_comparison changed to: {changed_parameter.value.string_value}")
                self.histogram_comparison = changed_parameter.value.string_value


def main(args=None):
    rclpy.init(args=args)
    node = HistogramFilter()
    node.load_scan_data()
    node.convert_scan_data_to_histograms()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
