import array
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ScanData:
    """
    Represents a scan data object.

    Attributes:
        _position (Tuple[float, float, float]): The position of the scan data.
        _measurements (np.ndarray): The measurements of the scan data.
    """

    _position: Tuple[float, float, float]
    _measurements: np.ndarray

    def __init__(self, position: Tuple[float, float, float], measurements: array.array):
        """
        Initializes a new instance of the ScanData class.

        Args:
            position (Tuple[float, float, float]): The position of the scan data.
            measurements (array.array): The measurements of the scan data.
        """
        self._position = position
        self._measurements = np.asarray(measurements, dtype=float)

    @property
    def position(self) -> Tuple[float, float, float]:
        """
        Gets the position of the scan data.

        Returns:
            Tuple[float, float, float]: The position of the scan data.
        """
        return self._position

    @property
    def measurements(self) -> np.ndarray:
        """
        Gets the measurements of the scan data.

        Returns:
            np.ndarray: The measurements of the scan data.
        """
        return self._measurements

    @measurements.setter
    def measurements(self, measurements: array.array):
        """
        Sets the measurements of the scan data.

        Args:
            measurements (array.array): The measurements of the scan data.
        """
        self._measurements = np.asarray(measurements, dtype=float)