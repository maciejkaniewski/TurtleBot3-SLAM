import os
from glob import glob


from setuptools import find_packages, setup

package_name = "utils"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "maps_pgm"), glob("maps_pgm/*.pgm")),
        (os.path.join("share", package_name, "maps_yaml"), glob("maps_yaml/*.yaml")),
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
            "map_scanner = utils.map_scanner:main",
        ],
    },
)
