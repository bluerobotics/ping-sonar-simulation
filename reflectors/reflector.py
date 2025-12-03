import numpy as np
from numpy.typing import NDArray


class SeaFloor:
    """
        Class to create seafloor topography. The seafloor topography is comprised of spherical reflectors. Each reflector's center is on the same plane.

        Attributes
        ----------

        depth : float
            depth of the seafloor in meters.

        topography : str
            specifies the desired seafloor topography. Currently only FLAT topographies supported.

        reflectance: float
            reflectance coefficient of each reflector

        seafloor: NDArray
            The grid of spheres. Each element of the array is a an array with the structure [x-coordinate, y-coordinate, z-coordinate, radius].

    """

    def __init__ (self, depth : float, reflectance : float = 0.5, topography: str = 'flat', min_size: float = 1, max_size: float = 10, num_spheres: int = 1E4):
        """
        Insantiates a SeaFloor object.

        Parameters
        ------
        depth: float
            depth of the seafloor in meters.

        reflectance: float
            reflectance coefficient of each reflector.

        topography: str
            specifies the desired seafloor topography. Currently only FLAT topographies supported.

        min_size: float
            minmum radius for the reflectors.

        max_size: float
            maximum radius for the reflectors.

        num_spheres: int
            number of spheres per side of grid.
        """

        self.depth = depth
        self.topography = topography
        self.reflectance = reflectance

        self.seafloor = self.create_seafloor(depth=depth, topography=topography, min_size=min_size, max_size=max_size, num_spheres=num_spheres)

    
    def create_seafloor(self, depth, topography: str = 'flat', min_size: float = 1, max_size: float = 5, num_spheres: int = 10000) -> NDArray:
        """
        Creates the seafloor. Always a 150m x 150m grid.

        Parameters
        ----------
        topography: str
            specifies the desired seafloor topography. Currently only FLAT topographies supported.

        min_size: float
            minmum radius for the reflectors.

        max_size: float
            maximum radius for the reflectors

        num_spheres: int
            number of columns and rows in grid of spheres

        Outputs
        -------
        seafloor: NDArray[Tuple[float, float, float]]
            An array of reflectors 
        """

        side_length = int(150E2) # Cone sweeps out 133 meter radius circle, so this fully encompasses the possible intersections with the transducer (assuming it's pointed straight down).

        seafloor = np.zeros(side_length)

        if (topography == 'flat'):
            x = np.random.uniform(-side_length / 2, side_length / 2, num_spheres)
            y = np.random.uniform(-side_length / 2, side_length / 2, num_spheres)
            z = np.full(num_spheres, -depth)
            
            # Random radii
            radii = np.random.uniform(min_size, max_size, num_spheres)
            
            # Stack into an array [x, y, r]
            seafloor = np.column_stack((x, y, z, radii))


        return seafloor





