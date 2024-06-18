import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import signal

# Define the path to your .pkl file
file_path = "turtlebot3_dqn_stage4_grid_0.25_full.pkl"
#file_path = "turtlebot3_dqn_stage4_grid_0.25_10_10.pkl"
#file_path = "turtlebot3_dqn_stage4_grid_0.25_5_5.pkl"
#file_path = "turtlebot3_dqn_stage4_grid_0.25_3_3.pkl"
file_path = "map11.pkl"
#file_path = "map15.pkl"

#file_path = "turtlebot3_dqn_stage4u_grid_0.25.pkl"

# Load the data from the .pkl file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# take only 5th element of data
#data = [data[7]]

# Occupancy grid map settings
RESOLUTION = 0.05
PROBABILITY = 0.55
MAX_PROBABILITY = 200
MAP_SIZE = (int(5/RESOLUTION), int(5/RESOLUTION))

class OccupancyGridMap:
    def __init__(self, map_size, resolution):
        self.map_size = map_size
        self.cell_size = resolution
        self.center_offset = np.array(self.map_size) / 2.0

        # Initialize the map with half probability
        initial_probability = 0.5
        log_odds_initial = math.log(initial_probability / (1 - initial_probability))

        self.map_data = np.full(self.map_size, 50, dtype=np.int8)  # Initialize with 50% probability
        self.map_probabilities = np.full(self.map_size, 50.0, dtype=np.float64)
        self.logaritmic_odds = np.full(self.map_size, log_odds_initial, dtype=np.float64)

    def update_probability(self, cell, logaritmic_odds: float) -> None:
        x, y = cell
        self.logaritmic_odds[y][x] += logaritmic_odds

        if self.logaritmic_odds[y][x] > MAX_PROBABILITY:
            self.logaritmic_odds[y][x] = MAX_PROBABILITY
        if self.logaritmic_odds[y][x] < -MAX_PROBABILITY:
            self.logaritmic_odds[y][x] = -MAX_PROBABILITY

        probability = 1.0 - 1.0 / (1.0 + math.exp(self.logaritmic_odds[y][x]))
        probability = int(probability * 100)

        self.map_probabilities[y][x] = probability
        self.map_data[y][x] = probability

    def find_cells(self, x0: int, y0: int, x1: int, y1: int) -> list:
        cells = []
        dx = x1 - x0
        dy = y1 - y0

        x, y = x0, y0

        step = max(abs(dx), abs(dy))

        if step == 0:
            return cells

        x_inc = dx / step
        y_inc = dy / step

        for _ in range(step):
            x += x_inc
            y += y_inc
            if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                cells.append([int(x), int(y)])

        return cells

    def update_map_with_scan(self, x, y, theta, laser_scan):
        # Filter out invalid laser scan data
        valid_indices = np.isfinite(laser_scan) & (laser_scan > 0)
        valid_laser_scan = laser_scan[valid_indices]

        if len(valid_laser_scan) == 0:
            return np.array([]), np.array([])  # No valid scan data to plot

        #Print position and theta
        print(f"Position: {x:2f}, {y:2f}, {theta:2f}")
        #diff = 90 - theta
        theta_rad = np.radians(theta)  # Convert theta to radians
        angles_deg = np.linspace(-180, 180, len(laser_scan))[valid_indices]  # Generate angles in degrees for 360 degrees laser scan
        angles_rad = np.radians(angles_deg +180)  # Convert to radians and apply 180-degree correction

        # Calculate the scan points' x and y coordinates
        scan_x = x + valid_laser_scan * np.cos(angles_rad + theta_rad)
        scan_y = y + valid_laser_scan * np.sin(angles_rad + theta_rad)

        # Convert world coordinates to grid coordinates
        grid_x = np.floor((scan_x / self.cell_size) + self.center_offset[0]).astype(int)
        grid_y = np.floor((scan_y / self.cell_size) + self.center_offset[1]).astype(int)

        # Convert robot's position from world coordinates to grid coordinates
        robot_x = int(x / self.cell_size + self.center_offset[0])
        robot_y = int(y / self.cell_size + self.center_offset[1])

        # Update occupancy probabilities
        for gx, gy in zip(grid_x, grid_y):
            if 0 <= gx < self.map_size[0] and 0 <= gy < self.map_size[1]:
                cells = self.find_cells(robot_x, robot_y, int(gx), int(gy))

                if not cells:
                    continue

                for cell in cells[:-1]:
                    self.update_probability(cell, -PROBABILITY)
                self.update_probability(cells[-1], PROBABILITY)

        return scan_x, scan_y

# Initialize the occupancy grid map
occupancy_grid_map = OccupancyGridMap(MAP_SIZE, RESOLUTION)

# Create subplots
fig, (ax_scan, ax_map) = plt.subplots(1, 2, figsize=(12, 6))

# Set up the scan plot
ax_scan.set_aspect('equal')
ax_scan.set_xlabel('X [m]')
ax_scan.set_ylabel('Y [m]')
ax_scan.set_xlim([-2.5, 2.5])
ax_scan.set_ylim([-2.5, 2.5])
ax_scan.set_title('Laser Scans')
ax_scan.grid(True)

# Set up the occupancy grid map plot
ax_map.set_aspect('equal')
ax_map.set_xlabel('X [m]')
ax_map.set_ylabel('Y [m]')
ax_map.set_title('Occupancy Grid Map')
ax_map.grid(True)

# Define a color cycle
colors = plt.cm.get_cmap('hsv', len(data))

# Flag to control the main loop
running = True

def handle_interrupt(signum, frame):
    global running
    running = False
    print("\nInterrupted! Exiting gracefully...")

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handle_interrupt)

try:
    for i, scan_data in enumerate(data):
        if not running:
            break

        x, y, theta = scan_data.position
        laser_scan = scan_data.measurements

        # Update the map with the current scan
        scan_x, scan_y = occupancy_grid_map.update_map_with_scan(x, y, theta, laser_scan)

        if len(scan_x) > 0 and len(scan_y) > 0:
            # Plot the robot's position and scan points
            ax_scan.plot(x, y, 'o', color=colors(i))  # Mark the robot's position with a red circle
            ax_scan.plot(scan_x, scan_y, '.', label=f'Scan {i + 1}', color=colors(i))

        # Update the occupancy grid map plot
        ax_map.imshow(
            occupancy_grid_map.map_data,
            cmap='gray_r',  # Invert the colormap for black obstacles and white free spaces
            extent=[
                -occupancy_grid_map.center_offset[0] * occupancy_grid_map.cell_size,
                occupancy_grid_map.center_offset[0] * occupancy_grid_map.cell_size,
                -occupancy_grid_map.center_offset[1] * occupancy_grid_map.cell_size,
                occupancy_grid_map.center_offset[1] * occupancy_grid_map.cell_size,
            ],
            alpha=0.5,
            origin='lower'
        )

        # Set plot limits for the map
        ax_map.set_xlim(-occupancy_grid_map.center_offset[0] * occupancy_grid_map.cell_size,
                        occupancy_grid_map.center_offset[0] * occupancy_grid_map.cell_size)
        ax_map.set_ylim(-occupancy_grid_map.center_offset[1] * occupancy_grid_map.cell_size,
                        occupancy_grid_map.center_offset[1] * occupancy_grid_map.cell_size)

        # Pause to allow viewing
        #plt.pause(0.50)

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Cleaning up and exiting...")
    #plt.close(fig)  # Close the figure window

# Display the final cumulative plot
plt.show(block=True)
