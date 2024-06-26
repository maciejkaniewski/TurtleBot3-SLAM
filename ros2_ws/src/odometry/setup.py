import os
from glob import glob

from setuptools import find_packages, setup

package_name = "odometry"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*_launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mkaniews",
    maintainer_email="maciejkaniewski1999@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "odom_vel = odometry.odom_vel:main",
            "odom_pos = odometry.odom_pos:main",
            "odom_errors = odometry.odom_errors:main",
        ],
    },
)
