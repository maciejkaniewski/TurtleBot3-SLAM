#!/usr/bin/env python3

import argparse
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml


class MapLoader:
    """
    A class for loading map data from YAML and PGM files.

    Args:
        yaml_file_path (str): The file path to the YAML metadata file.
        pgm_file_path (str): The file path to the PGM image file.

    Attributes:
        yaml_file_path (str): The file path to the YAML metadata file.
        pgm_file_path (str): The file path to the PGM image file.
        metadata (dict): The metadata loaded from the YAML file.
        img (numpy.ndarray): The image loaded from the PGM file.
        obstacle_coords (list): The coordinates of the obstacles in the map.

    """

    def __init__(self, yaml_file_path, pgm_file_path):
        self.yaml_file_path = yaml_file_path
        self.pgm_file_path = pgm_file_path
        self.metadata = self.load_yaml_metadata()
        self.img, self.obstacle_coords, self.obstacle_radius = self.load_map()

    def load_yaml_metadata(self):
        """
        Load the metadata from the YAML file.

        Returns:
            dict: The metadata loaded from the YAML file.

        """
        with open(self.yaml_file_path, "r") as file:
            metadata = yaml.safe_load(file)
        return metadata

    def load_map(self) -> None:
        """
        Load the map image and obstacle coordinates.

        Returns:
            tuple: A tuple containing the map image (numpy.ndarray) and the obstacle coordinates (list).

        """
        img = mpimg.imread(self.pgm_file_path)
        self.resolution = self.metadata["resolution"]
        self.origin = self.metadata["origin"]
        self.height, self.width = img.shape
        y_coords, x_coords = np.where(img < 250)
        map_x_coords = x_coords * self.resolution + self.origin[0]
        map_y_coords = (self.height - y_coords) * self.resolution + self.origin[1]
        map_x_coords = map_x_coords + self.resolution / 2
        map_y_coords = map_y_coords - self.resolution / 2
        obstacle_coords = list(zip(map_x_coords, map_y_coords))
        obstacle_radius = self.resolution / 2
        return img, obstacle_coords, obstacle_radius

    def get_obstacle_coordinates(self):
        """
        Get the coordinates of the obstacles in the map.

        Returns:
            list: The coordinates of the obstacles in the map.

        """
        return self.obstacle_coords

    def get_obstacle_radius(self):
        """
        Get the radius of the obstacles in the map.

        Returns:
            float: The radius of the obstacles in the map.

        """
        return self.obstacle_radius

    def visualize_map(self, laser_scan_data_path: str):
        """
        Visualizes the map with obstacles, unknown areas, and laser scans.
        """

        def load_laserscan_data(filename):
            with open(filename, "rb") as file:
                data = pickle.load(file)
            return data

        map_width = self.width * self.resolution
        map_height = self.height * self.resolution

        # Define extent of the image in the plot
        extent = [
            self.origin[0],
            self.origin[0] + map_width,
            self.origin[1],
            self.origin[1] + map_height,
        ]

        # Plotting the map
        plt.imshow(self.img, cmap="gray", extent=extent)
        plt.colorbar(label="Occupancy Level")
        plt.xlabel("X [meters]")
        plt.ylabel("Y [meters]")
        plt.title("ROS2 Map Visualization")

        y_coords, x_coords = np.where(self.img == 0)
        map_x_coords = x_coords * self.resolution + self.origin[0]
        map_y_coords = (self.height - y_coords) * self.resolution + self.origin[1]
        map_x_coords = map_x_coords + self.resolution / 2
        map_y_coords = map_y_coords - self.resolution / 2
        obstacles = plt.scatter( map_x_coords, map_y_coords, color="black", s=12, label="Obstacles")

        y_coords, x_coords = np.where(self.img == 205)
        map_x_coords = x_coords * self.resolution + self.origin[0]
        map_y_coords = (self.height - y_coords) * self.resolution + self.origin[1]
        map_x_coords = map_x_coords + self.resolution / 2
        map_y_coords = map_y_coords - self.resolution / 2
        unknown = plt.scatter(map_x_coords, map_y_coords, color="#cdcdcd", s=12, label="Unknown Area")

        # Load data from the provided pickle file
        laser_scan_data = load_laserscan_data(laser_scan_data_path)
        all_coords = np.array([data.position for data in laser_scan_data])
        laser_scans = plt.scatter(
            all_coords[:, 0],
            all_coords[:, 1],
            color="#98C379",
            s=12,
            label="Laser Scans",
        )

        plt.grid(True)
        plt.legend(handles=[obstacles, unknown, laser_scans], loc="upper right")
        plt.show()


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Visualize a PGM map with obstacles and LaserScan data.")
    parser.add_argument("pgm_file_path", help="The file path to the PGM image file.")
    parser.add_argument("yaml_file_path", help="The file path to the YAML metadata file.")
    parser.add_argument("laser_scan_data_path", help="The file path to the LaserScan data pickle file.")
    # fmt: on
    args = parser.parse_args()
    loader = MapLoader(args.yaml_file_path, args.pgm_file_path)
    loader.visualize_map(args.laser_scan_data_path)


if __name__ == "__main__":
    main()
