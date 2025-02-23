#!/usr/bin/env python3

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import rclpy
import tf_transformations
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterEvent
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from utils.scan_data import ScanData
from example_interfaces.srv import Trigger
from nav_msgs.msg import Odometry

INITIAL_POINTS = 0
THRESHOLD = 0.30

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
        self.declare_parameter('histogram_comparison', 'manhattan')
        self.declare_parameter('map_pkl_file', 'turtlebot3_dqn_stage4_grid_0.25_3_3.pkl')
        self.declare_parameter('plot_enabled', True)
        self.declare_parameter("odom_topic", "/odom_vel")

        # Get parameters
        self.histogram_bins = self.get_parameter('histogram_bins').get_parameter_value().integer_value
        self.histogram_range_m = self.get_parameter('histogram_range_m').get_parameter_value().double_array_value
        self.histogram_comparison = self.get_parameter('histogram_comparison').get_parameter_value().string_value
        self.map_pkl_file = self.get_parameter('map_pkl_file').get_parameter_value().string_value
        self.plot_enabled = self.get_parameter('plot_enabled').get_parameter_value().bool_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        

        self.histogram_range_m = tuple(self.histogram_range_m)

        # Log the parameters
        self.get_logger().info(f"histogram_bins: {self.histogram_bins}")
        self.get_logger().info(f"histogram_range_m: {self.histogram_range_m}")
        self.get_logger().info(f"histogram_comparison: {self.histogram_comparison}")
        self.get_logger().info(f"map_pkl_file: {self.map_pkl_file}")
        self.get_logger().info(f"plot_enabled: {self.plot_enabled}")
        self.get_logger().info(f"odom_topic: {self.odom_topic}")

        # Create /scan and /parameter_events subscriptions
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.event_subscription = self.create_subscription(ParameterEvent, '/parameter_events', self.parameter_event_callback, 10)
        self.odom_subscription = self.create_subscription(PoseStamped, self.odom_topic, self.odom_callback, 10)
        self.odom_true_subscription = self.create_subscription(Odometry, '/odom', self.odom_true_callback, 10)

        # Create /histogram_pose publisher with a 5 Hz timer
        self.hfliter_pose_publisher = self.create_publisher(PoseStamped, "/hfilter_pose", 10)
        self.hfilter_pose = self.create_timer(1 / 5, self.hfilter_pose_callback)

        # Create service for adding reference points
        self.srv = self.create_service(Trigger, 'trigger_hf', self.trigger_callback)
        self.pfilter_pose_subscription = self.create_subscription(PoseStamped, '/pfilter_pose', self.pfilter_pose_callback, 10)

        # Scan data
        self.scan_data = []
        self.current_scan_data = None
        self.closest_scan_data = None
        self.scan_data_histograms = np.array([])

        # Robot true position
        self.robot_true_x_m = 0.0
        self.robot_true_y_m = 0.0
        self.robot_true_theta_rad = 0.0

        # Robot estimated position with Histogram Filter
        self.robot_x_m = 0.0
        self.robot_y_m = 0.0
        self.robot_theta_rad = 0.0
        self.robot_theta_deg = 0.0

        # Robot estimated position with Particle Filter
        self.robot_pf_x_m = 0.0
        self.robot_pf_y_m = 0.0

        # Robot velocity/position position
        self.robot_x_m_o = 0.0
        self.robot_y_m_o = 0.0
        self.robot_theta_deg_o = 0.0

        # Probabilities for the Histogram Filter
        self.probabilities = []
        self.probabilties_coords = []

        # Plotting
        if self.plot_enabled:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
            self.fig.canvas.manager.set_window_title('Histogram Filter')
            self.plot_timer = self.create_timer(0.100, self.plot_callback)
            self.cbar_flag = True

    def pfilter_pose_callback(self, msg: PoseStamped) -> None:
        """
        Callback function for the particle filter pose subscriber.
        """

        self.robot_pf_x_m = msg.pose.position.x
        self.robot_pf_y_m = msg.pose.position.y

    def trigger_callback(self, request, response):
        """
        Callback function for handling trigger requests.
        """

        self.get_logger().info('Received a trigger request.')
        response.success = True
        response.message = 'Trigger for the Histogram Filter handled successfully.'
        if len(self.scan_data) <  INITIAL_POINTS:
            self.scan_data = np.append(self.scan_data, ScanData(position=(self.robot_x_m_o , self.robot_y_m_o , self.robot_theta_deg_o), measurements=self.current_scan_data))
            self.convert_scan_data_to_histograms()
            return response
        self.scan_data = np.append(self.scan_data, ScanData(position=(self.robot_pf_x_m, self.robot_pf_y_m, self.robot_theta_deg), measurements=self.current_scan_data))
        self.convert_scan_data_to_histograms()
        self.get_logger().info(f"Added a new reference point at: ({self.robot_pf_x_m:.2f}, {self.robot_pf_y_m:.2f})")
        return response

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
            # cap scan data to 3.5 meters
            scan_data.measurements = np.clip(scan_data.measurements, a_min=None, a_max=3.5)
            hist, _ = np.histogram(scan_data.measurements, range=self.histogram_range_m, bins=self.histogram_bins)
            new_scan_data = ScanData(position=scan_data.position, measurements=hist)
            self.scan_data_histograms = np.append(self.scan_data_histograms, new_scan_data)
        # fmt: on

    def compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compares two histograms using the specified method.
        """

        method = self.histogram_comparison
        if method == 'manhattan':
            return np.sum(np.abs(hist1 - hist2))
        if method == 'euclidean':
            return np.linalg.norm(hist1 - hist2)
        elif method == 'chi_square':
            return np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))
        
    def localize_robot(self, current_histogram: np.ndarray) -> tuple[float, float]:
        """
        Localizes the robot based on the current histogram.
        """

        estimated_x, estimated_y, index = 0, 0, 0

        self.probabilities = []
        self.probabilties_coords = []

        for i, scan_data in enumerate(self.scan_data_histograms):
            total_difference = self.compare_histograms(scan_data.measurements, current_histogram)
            
            # Calculate Euclidean distance from odometry position
            scan_x, scan_y, _ = scan_data.position
            distance = np.sqrt((scan_x - self.robot_x_m_o) ** 2 + (scan_y - self.robot_y_m_o) ** 2)
            
            # Only keep data points within the THRESHOLD distance
            if distance <= THRESHOLD:
                self.probabilities.append(total_difference)
                self.probabilties_coords.append(scan_data.position)

        # Convert lists to numpy arrays AFTER filtering
        self.probabilities = np.array(self.probabilities)
        self.probabilties_coords = np.array(self.probabilties_coords)

        if len(self.probabilities) == 0:
            self.get_logger().warn("No valid reference points within threshold! Localization might fail.")
            return self.robot_x_m_o, self.robot_y_m_o  # Return odometry position as fallback


        # Normalize probabilities to [0, 1] range
        self.probabilities = self.probabilities - np.min(self.probabilities)
        self.probabilities = self.probabilities / np.sum(self.probabilities)

        # Invert the probabilities and re-normalize to sum up to 1
        self.probabilities = (1 - self.probabilities) / np.sum(1 - self.probabilities)

        # LOG probabilities
        #self.get_logger().info(f"Probabilities: {self.probabilities}")

        index = np.argmax(self.probabilities)
        estimated_x, estimated_y, _ = self.probabilties_coords[index]
        # Assign closes scan data using the x,y coordinates
        
        min_distance = float('inf')
        for scan_data in self.scan_data:
            scan_x, scan_y, _ = scan_data.position
            distance = np.sqrt((scan_x - estimated_x) ** 2 + (scan_y - estimated_y) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                self.closest_scan_data = scan_data  # Assign the closest scan data

        #self.get_logger().info(f"Closest reference point: ({self.closest_scan_data.position[0]:.2f}, {self.closest_scan_data.position[1]:.2f})")


        return estimated_x, estimated_y

    def calculate_orientation(self, current_scan_data):
        """
        Calculates the orientation of the current scan data relative to the closest scan data.
        """

        ref_data = self.closest_scan_data.measurements

        # Adjust reference data to 0 degrees orientation by shifting the data
        # Assumed that the data was collected within 90 orientation, if not, adjust the shift value
        adjusted_ref_data = np.roll(ref_data, -90)
        # TODO: Adjust the shift value based on the reference data

        # Vectorize the computation of the sum of squared differences for all shifts
        diffs = np.array([
            np.sum((adjusted_ref_data - np.roll(current_scan_data, shift)) ** 2)
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

        self.current_scan_data = msg.ranges
        # Cap current scan data to 3.5 meters
        self.current_scan_data = np.clip(self.current_scan_data, a_min=None, a_max=3.5)
        hist, _ = np.histogram(msg.ranges, range=self.histogram_range_m, bins=self.histogram_bins)
        if len(self.scan_data) >= INITIAL_POINTS:
            self.robot_x_m, self.robot_y_m = self.localize_robot(hist)
            self.robot_theta_deg = self.calculate_orientation(self.current_scan_data)
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

    def check_and_reset_odometry(self):
        """
        Checks if the robot is close enough to the selected reference point to reset odometry.
        """

        RESET_THRESHOLD = 0.01  # Robot must be within 1cm of the reference point to reset odometry

        #  log closest_scan_data position
        if self.closest_scan_data is None:
            return
        
        # Use self.closest_scan_data to get the reference point
        distance = np.linalg.norm([self.robot_x_m_o - self.closest_scan_data.position[0], self.robot_y_m_o - self.closest_scan_data.position[1]])
        
        if distance < RESET_THRESHOLD:
            #self.get_logger().info("Odometry reset point reached!")
            self.robot_x_m_o = self.closest_scan_data.position[0]
            self.robot_y_m_o = self.closest_scan_data.position[1]
            #self.get_logger().info(f"Odometry reset to: ({self.robot_x_m_o:.2f}, {self.robot_y_m_o:.2f})")

    def odom_callback(self, msg: PoseStamped) -> None:
        """
        Callback function for the odometry velocity/position publisher.
        """

        self.robot_x_m_o = msg.pose.position.x
        self.robot_y_m_o = msg.pose.position.y
        orientation_q = msg.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
        self.robot_theta_deg_o = np.rad2deg(yaw)
        self.check_and_reset_odometry()

    
    def odom_true_callback(self, msg: Odometry) -> None:
        """
        Callback function for handling odometry messages.
        """

        self.robot_true_x_m = msg.pose.pose.position.x
        self.robot_true_y_m = msg.pose.pose.position.y

        # fmt: off
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w,]
        _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
        self.robot_true_theta_rad = yaw


    def plot_callback(self) -> None:
        """
        Callback function for plotting.

        """
        if len(self.scan_data) < INITIAL_POINTS:
            return

        self.ax.cla()
        # self.ax[1].cla()
        # self.ax[2].cla()

        # # Plot Current Scan Data
        # self.ax[0].plot(
        #     self.current_scan_data,
        #     label="Current LaserScan Data",
        #     color="#16FF00",
        #     linewidth=2,
        # )
        # self.ax[0].fill_between(
        #     range(len(self.current_scan_data)),
        #     self.current_scan_data,
        #     color="#16FF00",
        #     alpha=0.3
        # )

        # if self.closest_scan_data is None:
        #     return

        # self.ax[0].plot(
        #     self.closest_scan_data.measurements,
        #     label="Closest LaserScan Data",
        #     color="magenta",
        #     linewidth=2,
        #     linestyle='dashed',
        # )
        # self.ax[0].fill_between(
        #     range(len(self.closest_scan_data.measurements)),
        #     self.closest_scan_data.measurements,
        #     color="magenta",
        #     alpha=0.3
        # )
        # self.ax[0].set_xlabel("θ [°]", fontsize=12)
        # self.ax[0].set_ylabel("Distance [m]", fontsize=12)
        # self.ax[0].grid()
        # self.ax[0].legend()

        # # Plot the histogram of the current scan data
        # hist, bin_edges = np.histogram(self.current_scan_data, range=self.histogram_range_m, bins=self.histogram_bins)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # bin_width = bin_edges[1] - bin_edges[0]

        # self.ax[1].bar(bin_centers, hist, align="center", width=bin_width, edgecolor="#023EFF", color="#00ffff")

        # self.ax[1].set_xlabel("Distance [m]", fontsize=12)
        # self.ax[1].set_ylabel("Measurements Count", fontsize=12)
        # self.ax[1].set_xticks(bin_edges)
        # self.ax[1].tick_params(axis='both', which='major', labelsize=10)
        # self.ax[1].set_xlim(self.histogram_range_m)
        # self.ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        # self.ax[1].grid()

        # Plot the Probabilities
        scatter = self.ax.scatter(
            [x[0] for x in self.probabilties_coords],
            [x[1] for x in self.probabilties_coords],
            s=128,
            c=self.probabilities,
            cmap='gnuplot2_r',
            marker='s',
        )

        # Plot odometry robot position
        self.ax.scatter(self.robot_x_m_o, self.robot_y_m_o, color='orange', marker='o', s=150, label="Odometry Robot Position")
        # Plot a dashed circle around the odometry robot position
        circle = plt.Circle((self.robot_x_m_o, self.robot_y_m_o), THRESHOLD, color='blue', alpha=0.1, linestyle='dashed')
        self.ax.add_patch(circle)
        # annotate the odometry robot position
        self.ax.text(self.robot_x_m_o, self.robot_y_m_o, f'Odometry\n({self.robot_x_m_o:.2f},{self.robot_y_m_o:.2f})', fontsize=9, ha='left', color='orange')

        # plot robot true position
        self.ax.scatter(self.robot_true_x_m, self.robot_true_y_m, color='red', marker='o', s=150, label="True Robot Position")
        # annotate the true robot position
        self.ax.text(self.robot_true_x_m, self.robot_true_y_m, f'True\n({self.robot_true_x_m:.2f},{self.robot_true_y_m:.2f})', fontsize=9, ha='right', color='red')

        # Plot arrow of true robot orientation
        self.ax.quiver(self.robot_true_x_m, self.robot_true_y_m, np.cos(self.robot_true_theta_rad), np.sin(self.robot_true_theta_rad), color='red', scale=10, scale_units='xy', angles='xy')
        

        # Create a list of tuples with each probability and its corresponding coordinates
        probabilities_with_coords = list(zip(self.probabilities, self.probabilties_coords))

        # Sort the list of tuples by the probabilities in descending order
        sorted_probabilities_with_coords = sorted(probabilities_with_coords, key=lambda x: x[0], reverse=True)

        # # Iterate over the sorted list and annotate the points in the center
        for rank, (probability, (x, y, _)) in enumerate(sorted_probabilities_with_coords, start=1):
            self.ax.annotate(f"{rank}", (x, y), fontsize=10, color='#00ff00', ha='center', va='center')
            self.ax.text(x, y, f'({x:.2f},{y:.2f})', fontsize=9, ha='left', color='black')

        # Exclude refrence poitns that are not in the circle and mark them grey
        for scan in self.scan_data:
            distance = np.linalg.norm([self.robot_x_m_o - scan.position[0], self.robot_y_m_o - scan.position[1]])
            if distance > THRESHOLD:
                self.ax.scatter(scan.position[0], scan.position[1], color='grey', marker='s', s=100, label="Reference Points")

        if self.cbar_flag:
            cbar = plt.colorbar(scatter, ax=self.ax)
            cbar.set_label('Probability')
            cbar.set_ticks([]) 
            self.cbar_flag = False
            # Get the colorbar axis
            cbar_ax = cbar.ax
            
            # Add custom labels at the bottom and top of the colorbar
            cbar_ax.text(1.25, 0, 'Least\nProbable\nLocation', ha='left', va='bottom', transform=cbar_ax.transAxes)
            cbar_ax.text(1.25, 1, 'Most\nProbable\nLocation', ha='left', va='top', transform=cbar_ax.transAxes)
            

        self.ax.set_xlabel("X [m]",fontsize=14)
        self.ax.set_ylabel("Y [m]",fontsize=14)
        #set tic0k size
        self.ax.tick_params(axis='both', which='major', labelsize=12)
        # set plot ax2 to const size
        self.ax.set_xlim(-0.65, 0.65)
        self.ax.set_ylim(-0.65, 0.65)
        self.ax.grid()

        #Log odometry, hfilter and true robot positions
        self.get_logger().info(f"Odometry: ({self.robot_x_m_o:.2f}, {self.robot_y_m_o:.2f})")
        self.get_logger().info(f"Histogram Filter: ({self.robot_x_m:.2f}, {self.robot_y_m:.2f})")
        self.get_logger().info(f"True Robot: ({self.robot_true_x_m:.2f}, {self.robot_true_y_m:.2f})")
        
        plt.draw()
        plt.pause(0.00001)


def main(args=None):
    rclpy.init(args=args)
    node = HistogramFilter()
    if INITIAL_POINTS == 0:
        node.load_scan_data()
        node.convert_scan_data_to_histograms()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()


if __name__ == "__main__":
    main()
