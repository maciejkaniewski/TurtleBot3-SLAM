from dataclasses import dataclass


@dataclass
class Particle:
    """Represents a particle".

    Attributes:
        x (float): The x-coordinate of the particle.
        y (float): The y-coordinate of the particle.
        theta (float): The orientation angle of the particle.
        weight (float): The weight of the particle.
    """

    x: float
    y: float
    theta: float
    weight: float

    def __post_init__(self):
        
        self.weight = self.weight

    @property
    def x(self):
        """Get the x-coordinate of the particle."""
        return self._x

    @x.setter
    def x(self, value):
        """Set the x-coordinate of the particle."""
        self._x = value

    @property
    def y(self):
        """Get the y-coordinate of the particle."""
        return self._y

    @y.setter
    def y(self, value):
        """Set the y-coordinate of the particle."""
        self._y = value

    @property
    def theta(self):
        """Get the orientation angle of the particle."""
        return self._theta

    @theta.setter
    def theta(self, value):
        """Set the orientation angle of the particle."""
        self._theta = value

    @property
    def weight(self):
        """Get the weight of the particle."""
        return self._weight

    @weight.setter
    def weight(self, value):
        """Set the weight of the particle.

        Args:
            value (float): The weight value.

        Raises:
            ValueError: If the weight is not between 0 and 1.

        """
        if not (0 <= value <= 1):
            raise ValueError("Weight must be between 0 and 1")
        self._weight = value
