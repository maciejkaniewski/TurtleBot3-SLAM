#!/usr/bin/env python3

import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy
import scipy.stats
import tf_transformations
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from particle_filter.particle import Particle
#from particle import Particle # for debugging
from rcl_interfaces.msg import ParameterEvent
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from utils.map_loader import MapLoader
from utils.scan_data import ScanData

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
        self.declare_parameter("theta_range_deg", [0, 360])
        self.declare_parameter("odom_topic", "/odom_vel")
        self.declare_parameter("predict_noise", [0.015,0.005 ])
        self.declare_parameter("update_noise", 0.125)
        self.declare_parameter("update_method", 'gaussian')
        self.declare_parameter("resampling_method", 'systematic')
        self.declare_parameter('map_pkl_file', 'turtlebot3_dqn_stage4_grid_0.25_3_3.pkl')
        self.declare_parameter('map_pgm_file', 'turtlebot3_dqn_stage4.pgm')
        self.declare_parameter('map_yaml_file', 'turtlebot3_dqn_stage4.yaml')
        self.declare_parameter('plot_enabled', True)
        
        # Get parameters
        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value
        self.x_range_m = tuple(self.get_parameter("x_range_m").get_parameter_value().double_array_value)
        self.y_range_m = tuple(self.get_parameter("y_range_m").get_parameter_value().double_array_value)
        self.theta_range_deg = tuple(self.get_parameter("theta_range_deg").get_parameter_value().integer_array_value)
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.predict_noise = tuple(self.get_parameter("predict_noise").get_parameter_value().double_array_value)
        self.update_noise = self.get_parameter("update_noise").get_parameter_value().double_value
        self.update_method = self.get_parameter("update_method").get_parameter_value().string_value
        self.resampling_method = self.get_parameter("resampling_method").get_parameter_value().string_value
        self.map_pkl_file = self.get_parameter('map_pkl_file').get_parameter_value().string_value
        self.map_pgm_file = self.get_parameter('map_pgm_file').get_parameter_value().string_value
        self.map_yaml_file = self.get_parameter('map_yaml_file').get_parameter_value().string_value
        self.plot_enabled = self.get_parameter('plot_enabled').get_parameter_value().bool_value

        # Log the parameters
        self.get_logger().info(f"num_particles: {self.num_particles}")
        self.get_logger().info(f"x_range_m: {self.x_range_m}")
        self.get_logger().info(f"y_range_m: {self.y_range_m}")
        self.get_logger().info(f"theta_range_deg: {self.theta_range_deg}")
        self.get_logger().info(f"odom_topic: {self.odom_topic}")
        self.get_logger().info(f"predict_noise: {self.predict_noise}")
        self.get_logger().info(f"update_noise: {self.update_noise}")
        self.get_logger().info(f"update_method: {self.update_method}, allowed values: ['simple', 'gaussian']")
        self.get_logger().info(f"resampling_method: {self.resampling_method}, allowed values: ['multinomial', 'systematic']")
        self.get_logger().info(f"map_pkl_file: {self.map_pkl_file}")
        self.get_logger().info(f"map_pgm_file: {self.map_pgm_file}")
        self.get_logger().info(f"map_yaml_file: {self.map_yaml_file}")
        self.get_logger().info(f"plot_enabled: {self.plot_enabled}")
        
        # Create required subscriptions
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.odom_true_subscription = self.create_subscription(Odometry, '/odom', self.odom_true_callback, 10)
        self.odom_subscription = self.create_subscription(PoseStamped, self.odom_topic, self.odom_callback, 10)
        self.hfilter_pose_subscription = self.create_subscription(PoseStamped, '/hfilter_pose', self.hfilter_pose_callback, 10)
        self.event_subscription = self.create_subscription(ParameterEvent, '/parameter_events', self.parameter_event_callback, 10)

        # Create /particles_poses publisher with a 30 Hz timer
        self.particles_poses_publisher = self.create_publisher(PoseArray, "/particles_poses", 10)
        self.timer_particles_poses = self.create_timer(1 / 30, self.particles_poses_callback)

        # Create /reference_points_poses publisher with a 1Hz timer
        self.reference_points_poses_publisher = self.create_publisher(PoseArray, "/reference_points_poses", 10)
        self.timer_reference_points_poses = self.create_timer(1, self.reference_points_poses_callback)

        # Create /particle_pose publisher with a 30 Hz timer
        self.pfliter_pose_publisher = self.create_publisher(PoseStamped, "/pfilter_pose", 10)
        self.timer_pfilter_pose = self.create_timer(1 / 30, self.pfilter_pose_callback)

        # Create /pgm_map publisher for visualization
        self.pgm_map_publisher = self.create_publisher(OccupancyGrid, "/pgm_map", 10)
        self.load_pgm_map()
        self.publish_pgm_map()
        self.destroy_publisher(self.pgm_map_publisher)

        # Create particles
        self.particles = self.initialize_particles(self.num_particles, self.x_range_m, self.y_range_m, self.theta_range_deg)
        self.particles_poses = self.particles_to_poses(self.particles)

        # Initialize the previous odometry pose
        self.previous_odom_pose = PoseStamped()
        self.previous_odom_pose_initialized = False

        # Reference points
        self.reference_points = self.load_refernece_points()
        self.closest_refernece_point = ScanData(position=(0, 0, 0), measurements=[0] * 360)

        # Scan data
        self.current_scan_data = None
        self.aligned_current_scan_data = None
        self.orientation_difference = 0

        # Distances to reference points
        self.displacements_x = []
        self.displacements_y = []
        self.euclidean_distances = np.zeros(len(self.reference_points))
        self.true_euclidean_distances = np.zeros(len(self.reference_points))

        # Robot true position
        self.robot_true_x_m = 0.0
        self.robot_true_y_m = 0.0
        self.robot_true_theta_rad = 0.0

        # Robot estimated position
        self.robot_x_m = 0.0
        self.robot_y_m = 0.0

        # Timer for logging
        self.timer_logger = self.create_timer(0.1, self.logger_callback)

        # Plotting
        if self.plot_enabled:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
            self.fig.canvas.manager.set_window_title('Particle Filter')
            self.plot_timer = self.create_timer(1, self.plot_callback)
            self.cbar_flag = True
    
    def load_pgm_map(self) -> None:
        """
        Loads the map from the specified PGM and YAML files.
        """

        world_pgm_path = os.path.join(
            get_package_share_directory("utils"),
            "maps_pgm",
            self.map_pgm_file
        )

        world_yaml_path = os.path.join(
            get_package_share_directory("utils"),
            "maps_yaml",
            self.map_yaml_file
        )

        self.pgm_map = MapLoader(world_yaml_path, world_pgm_path)

    def publish_pgm_map(self) -> None:
        """
        Publishes the loaded map as an OccupancyGrid message.
        """

        time.sleep(2)
        occupancy_grid_msg = OccupancyGrid()
        inverted_img = 255 - self.pgm_map.img
        scaled_img = np.flip((inverted_img * (100.0 / 255.0)).astype(np.int8), 0)
        self.map_data = scaled_img.ravel()

        occupancy_grid_msg.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id="odom"
        )
        occupancy_grid_msg.info.width = self.pgm_map.width
        occupancy_grid_msg.info.height = self.pgm_map.height
        occupancy_grid_msg.info.resolution = self.pgm_map.resolution
        occupancy_grid_msg.info.origin = Pose()
        occupancy_grid_msg.info.origin.position.x = self.pgm_map.origin[0]
        occupancy_grid_msg.info.origin.position.y = self.pgm_map.origin[1]
        occupancy_grid_msg.info.origin.position.z = 0.0
        occupancy_grid_msg.info.origin.orientation.w = 1.0
        occupancy_grid_msg.data = self.map_data.tolist()
        self.pgm_map_publisher.publish(occupancy_grid_msg)
    
    def load_refernece_points(self):
        """
        Loads scan data from a pickle file.
        """

        scan_data_path = os.path.join(
            get_package_share_directory("utils"),
            "maps_data",
            self.map_pkl_file,
        )

        with open(scan_data_path, "rb") as file:
            reference_points = pickle.load(file)
            return reference_points

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
    
    def predict_particles(self, particles: list, pose_msg: PoseStamped, std=(0.015, 0.005)) -> list:
        """
        Predicts the particles based on the odometry data.

        Args:
            particles (list): The list of particles.
            pose_msg (PoseStamped): The odometry data.
            std (tuple, optional): The standard deviation of the odometry data. 
        """

        pose_std, angle_std = std
        
        _, _, previous_yaw = tf_transformations.euler_from_quaternion([
            self.previous_odom_pose.pose.orientation.x,
            self.previous_odom_pose.pose.orientation.y,
            self.previous_odom_pose.pose.orientation.z,
            self.previous_odom_pose.pose.orientation.w
        ])

        _, _, current_yaw = tf_transformations.euler_from_quaternion([
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        ])

        delta_theta = current_yaw - previous_yaw
        delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta)) 

        delta_x = pose_msg.pose.position.x - self.previous_odom_pose.pose.position.x
        delta_y = pose_msg.pose.position.y - self.previous_odom_pose.pose.position.y
        distance_travelled = np.linalg.norm([delta_x, delta_y])

        for particle in particles:
            particle.x += distance_travelled * np.cos(particle.theta) + np.random.normal(0, pose_std)
            particle.y += distance_travelled * np.sin(particle.theta) + np.random.normal(0, pose_std)
            particle.theta += delta_theta + np.random.normal(0, angle_std)
            particle.theta = np.arctan2(np.sin(particle.theta), np.cos(particle.theta))

        self.previous_odom_pose = pose_msg

    def update_particles(self, particles, z, landmarks, R=0.05, method='simple'):
        """
        Update particles' weights based on measurement z, noise R, and landmarks.

        Args:
            particles (list of Particle): The particles to update.
            z (np.array): Array of measurements.
            R (float): Measurement noise.
            landmarks (list of tuples): Positions of landmarks.
        """

        weights = np.array([particle.weight for particle in particles])
        for i, landmark in enumerate(landmarks):
            distances = np.array(
                [
                    np.linalg.norm([particle.x - landmark.position[0], particle.y - landmark.position[1]])
                    for particle in particles
                ]
            )

            if method== 'simple':
                errors = np.abs(distances - z[i])
                weights *= 1 / (errors + 1.0e-6) 
            elif method == 'gaussian':
                weights *= scipy.stats.norm(distances, R).pdf(z[i])

        weights += 1.0e-300 
        weights /= np.sum(weights)

        for i, particle in enumerate(particles):
            particle.weight = weights[i]
    
    def neff(self, particles):
        """
        Calculate the effective number of particles (N_eff), based on their weights.
        """

        weights = np.array([particle.weight for particle in particles])
        return 1.0 / np.sum(np.square(weights))
    
    def multinomail_resampling(self, particles):
        """
        Performs multinomial resampling on a list of particles.
        """

        N = len(particles)
        weights = np.array([particle.weight for particle in particles])
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.
        indexes = np.searchsorted(cumulative_sum, np.random.random(N))

        new_particles = [particles[index] for index in indexes]
        resampled_particles = [
            Particle(p.x, p.y, p.theta, 1.0 / len(indexes)) for p in new_particles
        ]

        return resampled_particles

    def systematic_resampling(self, particles):
        """
        Performs systematic resampling on a list of Particle objects, returning only the indexes.
        """

        N = len(particles)
        weights = np.array([particle.weight for particle in particles])

        positions = (np.random.random() + np.arange(N)) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        return indexes

    def resample_from_index(self, particles, indexes):
        """
        Resample the list of particles according to the provided indexes, adjusting weights.
        """

        new_particles = [particles[index] for index in indexes]
        resampled_particles = [
            Particle(p.x, p.y, p.theta, 1.0 / len(indexes)) for p in new_particles
        ]

        return resampled_particles
    
    def estimate_particles(self, particles):
        """
        Estimates the mean and variance of the particle positions.
        """

        pos = np.array([[p._x, p._y] for p in particles])
        weights = np.array([p._weight for p in particles])
        mean = np.average(pos, weights=weights, axis=0)
        var = np.average((pos - mean) ** 2, weights=weights, axis=0)
        return mean, var

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
    
    def reference_points_to_poses(self, reference_points: list) -> PoseArray:

        poses = PoseArray()
        poses.header.frame_id = "odom"

        reference_points_positions = [point.position for point in reference_points]

        for point in reference_points_positions:
            pose = Pose()
            pose.position.x = point[0]
            pose.position.y = point[1]
            quaternion = tf_transformations.quaternion_from_euler(0, 0, np.deg2rad(point[2]))
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]
            poses.poses.append(pose)
        return poses
    
    def get_closest_reference_point(self, msg: PoseStamped, threshold: float) -> None:
        """
        Finds the reference point based on the given pose.

        Args:
            msg (PoseStamped): The pose message.
            threshold (float): The threshold for matching the reference point.
        """
        for point in self.reference_points:
            if abs(point.position[0] - msg.pose.position.x) < threshold and abs(point.position[1] - msg.pose.position.y) < threshold:
                return point

    def get_indices_around(self, reference_index, num_indices):
        """
        Returns a list of indices centered around a reference index.
        """

        indices = []

        for i in range(-num_indices // 2, num_indices // 2 + 1):
            indices.append((reference_index + i) % (359 + 1))

        return indices
    
    def calculate_scan_displacement(self, reference_scan, aligned_scan, axis='x', cone_angle=10):
        """
        Calculates the displacement between reference_scan and aligned_scan in the specified axis.
        """

        if axis == 'x':
            angles = [90, 270]  # Angles for x-axis displacement
        elif axis == 'y':
            angles = [359, 180]   # Angles for y-axis displacement
        else:
            raise ValueError("Axis must be either 'x' or 'y'")

        min_distances = []
        for angle in angles:
            indx = self.get_indices_around(angle, cone_angle)
            distances = [np.abs(reference_scan[i] - aligned_scan[i]) for i in indx]
            min_distances.append(min(distances))

        avg_dist = sum(min_distances) / len(min_distances)
        return avg_dist

    def particles_poses_callback(self) -> None:
        """
        Publishes the particle poses.
        """
        self.particles_poses = self.particles_to_poses(self.particles)
        self.particles_poses_publisher.publish(self.particles_poses)

    def reference_points_poses_callback(self) -> None:
        """
        Publishes the reference points poses.
        """
        self.reference_points_poses = self.reference_points_to_poses(self.reference_points)
        self.reference_points_poses_publisher.publish(self.reference_points_poses)

    def pfilter_pose_callback(self) -> None:
        """
        Callback function for the particle filter pose publisher.
        """

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "odom"

        pose = pose_stamped.pose
        pose.position.x = self.robot_x_m
        pose.position.y = self.robot_y_m

        quaternion = tf_transformations.quaternion_from_euler(0, 0, np.deg2rad(self.orientation_difference))
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        self.pfliter_pose_publisher.publish(pose_stamped)

    def odom_callback(self, msg: PoseStamped) -> None:
        """
        Callback function for the odometry velocity/position publisher.
        """

        if self.previous_odom_pose_initialized is False:
            self.previous_odom_pose = msg
            self.previous_odom_pose_initialized = True

        self.predict_particles(self.particles, msg, self.predict_noise)

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

    def hfilter_pose_callback(self, msg: PoseStamped) -> None:
        """
        Callback function for the hfilter pose publisher.
        """

        orientation_q = msg.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
        self.orientation_difference = np.rad2deg(yaw)
        self.closest_refernece_point = self.get_closest_reference_point(msg, threshold=0.05)

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback function for handling laser scan messages.

        Args:
            msg (LaserScan): The incoming laser scan message.
        """

        self.current_scan_data = msg.ranges

        # Replace 'inf' values with the maximum valid value from the reference data
        max_valid_value = np.nanmax(self.closest_refernece_point.measurements[~np.isinf(self.closest_refernece_point.measurements)])  # Maximum value among non-infinite values.
        current_scan_data_replaced_inf = np.where(np.isinf(self.current_scan_data), max_valid_value, self.current_scan_data)
        self.current_scan_data = current_scan_data_replaced_inf

        # Aligin the current scan data with the closest reference point
        self.aligned_current_scan_data = np.roll(self.current_scan_data, int(self.orientation_difference - self.closest_refernece_point.position[2]))

        self.displacements_x = []
        self.displacements_y = []
        self.euclidean_distances = []
        self.true_euclidean_distances = []

        #Calulate displcements for every refenece point
        for point in self.reference_points:
            x_displacement = self.calculate_scan_displacement(point.measurements, self.aligned_current_scan_data, axis='x')
            y_displacement = self.calculate_scan_displacement(point.measurements, self.aligned_current_scan_data, axis='y')
            self.displacements_x.append(x_displacement)
            self.displacements_y.append(y_displacement)

            euclidean_distance = np.linalg.norm([x_displacement, y_displacement])
            self.euclidean_distances.append(euclidean_distance)

            euclidean_distance_true = np.linalg.norm([self.robot_true_x_m - point.position[0], self.robot_true_y_m - point.position[1]])
            self.true_euclidean_distances.append(euclidean_distance_true)

        self.update_particles(self.particles, self.euclidean_distances, self.reference_points, self.update_noise, self.update_method)
        self.particles_poses = self.particles_to_poses(self.particles)
        self.particles_poses_publisher.publish(self.particles_poses)
        if self.neff(self.particles) < len(self.particles) / 2:
            if self.resampling_method == 'multinomial':
                self.particles = self.multinomail_resampling(self.particles)
            elif self.resampling_method == 'systematic':
                indexes = self.systematic_resampling(self.particles)
                self.particles = self.resample_from_index(self.particles, indexes)
            assert np.allclose(
                np.array([p.weight for p in self.particles]),
                1 / len(self.particles),
            )
        mean, var = self.estimate_particles(self.particles)
        self.robot_x_m = mean[0]
        self.robot_y_m = mean[1]

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
                self.previous_odom_pose_initialized = False
                self.odom_subscription = self.create_subscription(PoseStamped, self.odom_topic, self.odom_callback, 10)
            elif changed_parameter.name == "predict_noise":
                self.get_logger().info(f"predict_noise changed to: {changed_parameter.value.double_array_value}")
                self.predict_noise = tuple(changed_parameter.value.double_array_value)
            elif changed_parameter.name == "update_noise":
                self.get_logger().info(f"update_noise changed to: {changed_parameter.value.double_value}")
                self.update_noise = changed_parameter.value.double_value
            elif changed_parameter.name == "update_method":
                self.get_logger().info(f"update_method changed to: {changed_parameter.value.string_value}")
                self.update_method = changed_parameter.value.string_value
            elif changed_parameter.name == "resampling_method":
                self.get_logger().info(f"resampling_method changed to: {changed_parameter.value.string_value}")
                self.resampling_method = changed_parameter.value.string_value

    def plot_callback(self) -> None:
        """
        Callback function for plotting.

        """

        self.ax.cla()

        #Plot Current Scan Data
        self.ax.plot(
            self.current_scan_data,
            label="Current LaserScan Data",
            color="#16FF00",
            linewidth=2,
        )
        self.ax.fill_between(
            range(len(self.current_scan_data)),
            self.current_scan_data,
            color="#16FF00",
            alpha=0.3
        )

        self.ax.plot(
            self.closest_refernece_point.measurements,
            label="Closest LaserScan Data",
            color="magenta",
            linewidth=2,
            linestyle='dashed',
        )
        self.ax.fill_between(
            range(len(self.closest_refernece_point.measurements)),
            self.closest_refernece_point.measurements,
            color="magenta",
            alpha=0.3
        )

        # self.ax.plot(
        #     self.aligned_current_scan_data,
        #     label="Aligned Current LaserScan Data",
        #     color="blue",
        #     linewidth=2,
        #     linestyle='dashed',
        # )
        # self.ax.fill_between(
        #     range(len( self.aligned_current_scan_data)),
        #     self.aligned_current_scan_data,
        #     color="blue",
        #     alpha=0.3
        # )

        self.ax.set_xlabel("θ [°]", fontsize=16)
        self.ax.set_ylabel("Distance [m]", fontsize=16)
        self.ax.tick_params(axis='both', which='major', labelsize=14)
        #set legend font size
        self.ax.grid()
        self.ax.legend(fontsize=14, loc='upper right')

        plt.draw()
        plt.pause(0.00001)

    def logger_callback(self) -> None:
        """
        Callback function for logging information.
        """

        # for i, point in enumerate(self.reference_points):
        #     self.get_logger().info(
        #         f"Reference Point {i}: ({point.position[0]:>5.2f}, {point.position[1]:>5.2f}, {point.position[2]:>5.2f}) "
        #         f"Euclidean Distance: {self.euclidean_distances[i]:.2f}, True Euclidean Distance: {self.true_euclidean_distances[i]:.2f}"
        #     )

        # #log particles positions with theri weights
        # for i, particle in enumerate(self.particles):
        #     self.get_logger().info(f"Particle {i}: ({particle.x:.2f}, {particle.y:.2f}, {np.degrees(particle.theta):.2f}) Weight: {particle.weight:.2f}")



def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()


if __name__ == "__main__":
    main()
