#!/usr/bin/env python3

import os
import pickle
import time

import numpy as np
import rclpy
import tf_transformations
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid
from particle_filter.particle import Particle
#from particle import Particle # for debugging
from rcl_interfaces.msg import ParameterEvent
from rclpy.node import Node
from std_msgs.msg import Header
from utils.map_loader import MapLoader


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
        self.declare_parameter("odom_std", [0.0, 360.0])
        self.declare_parameter('map_pkl_file', 'turtlebot3_dqn_stage4_grid_0.25_3_3.pkl')
        self.declare_parameter('map_pgm_file', 'turtlebot3_dqn_stage4.pgm')
        self.declare_parameter('map_yaml_file', 'turtlebot3_dqn_stage4.yaml')
        

        # Get parameters
        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value
        self.x_range_m = tuple(self.get_parameter("x_range_m").get_parameter_value().double_array_value)
        self.y_range_m = tuple(self.get_parameter("y_range_m").get_parameter_value().double_array_value)
        self.theta_range_deg = tuple(self.get_parameter("theta_range_deg").get_parameter_value().integer_array_value)
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.odom_std = tuple(self.get_parameter("odom_std").get_parameter_value().double_array_value)
        self.map_pkl_file = self.get_parameter('map_pkl_file').get_parameter_value().string_value
        self.map_pgm_file = self.get_parameter('map_pgm_file').get_parameter_value().string_value
        self.map_yaml_file = self.get_parameter('map_yaml_file').get_parameter_value().string_value

        # Log the parameters
        self.get_logger().info(f"num_particles: {self.num_particles}")
        self.get_logger().info(f"x_range_m: {self.x_range_m}")
        self.get_logger().info(f"y_range_m: {self.y_range_m}")
        self.get_logger().info(f"theta_range_deg: {self.theta_range_deg}")
        self.get_logger().info(f"odom_topic: {self.odom_topic}")
        self.get_logger().info(f"odom_std: {self.odom_std}")
        self.get_logger().info(f"map_pkl_file: {self.map_pkl_file}")
        self.get_logger().info(f"map_pgm_file: {self.map_pgm_file}")
        self.get_logger().info(f"map_yaml_file: {self.map_yaml_file}")
        

        # Create /parameter_events and selected odom topic subscription
        self.event_subscription = self.create_subscription(ParameterEvent, '/parameter_events', self.parameter_event_callback, 10)
        self.hfilter_pose_subscription = self.create_subscription(PoseStamped, '/hfilter_pose', self.hfilter_pose_callback, 10)
        self.odom_subscription = self.create_subscription(PoseStamped, self.odom_topic, self.odom_callback, 10)

        # Create /particles_poses publisher with a 30 Hz timer
        self.particles_poses_publisher = self.create_publisher(PoseArray, "/particles_poses", 10)
        self.timer_particles_poses = self.create_timer(1 / 30, self.particles_poses_callback)

        # Create /reference_points_poses publisher with a 1Hz timer
        self.reference_points_poses_publisher = self.create_publisher(PoseArray, "/reference_points_poses", 10)
        self.timer_reference_points_poses = self.create_timer(1, self.reference_points_poses_callback)

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
    
    def predict_particles(self, particles: list, pose_msg: PoseStamped, std=(0.0, 0.0)) -> list:
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

    def odom_callback(self, msg: PoseStamped) -> None:
        """
        Callback function for the odometry velocity publisher.
        """

        if self.previous_odom_pose_initialized is False:
            self.previous_odom_pose = msg
            self.previous_odom_pose_initialized = True

        self.predict_particles(self.particles, msg, self.odom_std)

    def hfilter_pose_callback(self, msg: PoseStamped) -> None:
        """
        Callback function for the hfilter pose publisher.
        """

        # Log revieved pf filter pose
        self.get_logger().info(f"(X: {msg.pose.position.x:.2f} [m], Y: {msg.pose.position.y:.2f} [m], θ: {msg.pose.position.z:.2f} [°])")
        # having the pos x,y, and theta i want to get the measuremnt from reference points with this coords
        
        threshold = 0.1

        for point in self.reference_points:
            if abs(point.position[0] - msg.pose.position.x) < threshold and abs(point.position[1] - msg.pose.position.y) < threshold:
                self.get_logger().info(f"Reference point found: {point.position}")
                break

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
