import numpy as np

def generate_positions(n_circles, max_angle = 45, zero_dim = "z"):
    """ 
    The function generates random positions for n_circles circles. The circles are placed in a 3D space, but the
    geometry is constrained to a 2D plane. The function returns the positions of the circles and their radii.

    n_circles: number of circles
    max_angle: maximum angle between the position vectors of two consecutive circles (in degrees)
    zero_dim: the dimension in which the circles are constrained. It can be "x", "y", or "z"
    """

    dis = np.random.uniform(0.1, 1, n_circles)
    angle_in_radian = max_angle * np.pi / 180
    thetas = np.random.uniform(angle_in_radian/2, angle_in_radian, n_circles)

    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    if zero_dim.lower() == "z":
        x = (dis * cos_theta)
        y = (dis * sin_theta)
        z = np.zeros(n_circles)
    elif zero_dim.lower() == "y":
        x = (dis * cos_theta)
        z = (dis * sin_theta)
        y = np.zeros(n_circles)
    elif zero_dim.lower() == "x":
        y = (dis * cos_theta)
        z = (dis * sin_theta)
        x = np.zeros(n_circles)
    else:
        raise ValueError("zero_dim must be 'x', 'y', or 'z'")

    abs_pos = np.column_stack((x, y, z)) # Absolute positions of the circles
    rel_pos = np.cumsum(abs_pos, axis=0) # Relative positions of the circles w.r.t. the previous circle


    # Calculate Euclidean distances between consecutive positions
    euclidean_distances = np.sqrt(np.sum(np.diff(rel_pos, axis=0)**2, axis=1))

    # Calculate max allowable radius for each circle
    max_rads = np.zeros(n_circles)
    for i in range(1, n_circles - 1):
        max_rads[i] = min(euclidean_distances[i-1], euclidean_distances[i]) / 2
    max_rads[0] = euclidean_distances[0] / 2
    max_rads[-1] = euclidean_distances[-1] / 2

    rad = np.random.uniform(max_rads*.2 , max_rads)

    return rel_pos, rad
