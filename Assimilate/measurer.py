import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


def bresenham_line_2d(x0, y0, x1, y1):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    :param x0: The start x coordinate
    :param y0: The start y coordinate
    :param x1: The end x coordinate
    :param y1: The end y coordinate
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))  # Add the end point
    return points


def bresenham_line_3d(x0, y0, z0, x1, y1, z1):
    """Bresenham's Line Algorithm in 3D
    Produces a list of tuples from start and end in 3D space

    :param x0: The start x coordinate
    :param y0: The start y coordinate
    :param z0: The start z coordinate
    :param x1: The end x coordinate
    :param y1: The end y coordinate
    :param z1: The end z coordinate
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    xs = -1 if x0 > x1 else 1
    ys = -1 if y0 > y1 else 1
    zs = -1 if z0 > z1 else 1

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:        
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x0 != x1:
            points.append((x0, y0, z0))
            x0 += xs
            if p1 >= 0:
                y0 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z0 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz

    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:        
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y0 != y1:
            points.append((x0, y0, z0))
            y0 += ys
            if p1 >= 0:
                x0 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z0 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz

    # Driving axis is Z-axis
    else:        
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z0 != z1:
            points.append((x0, y0, z0))
            z0 += zs
            if p1 >= 0:
                y0 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x0 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            
    points.append((x0, y0, z0))  # Add the end point
    return points


def simulate_ray_2d(nx, ny, shooter, receiver):
    """Draws a line in a matrix from shooter to end using Bresenham's algorithm

    :param matrix: 2D numpy array
    :param shooter: Tuple of (x, y) coordinates for the start point
    :param receiver: Tuple of (x, y) coordinates for the end point
    """
    matrix = np.zeros((nx, ny))
    for point in bresenham_line_2d(shooter[0], shooter[1], receiver[0], receiver[1]):
        matrix[point[1], point[0]] = 1  # Note the order of indices (y, x) for numpy array
    return matrix


def simulate_ray_3d(nx, ny, nz, shooter, receiver):
    """
    Simulate the path of a ray shot from a shooter to a receiver in a 3D volume.
    The shooter and receiver can be at any position within the volume.
    
    :param nx: The width of the 3D volume.
    :param ny: The height of the 3D volume.
    :param nz: The depth of the 3D volume.
    :param shooter: Tuple (x, y, z) for the shooter's position.
    :param receiver: Tuple (x, y, z) for the receiver's position.
    """

    matrix = np.zeros((nx, ny, nz))

    # Path from the shooter to the receiver
    complete_path = bresenham_line_3d(shooter[0], shooter[1], shooter[2], receiver[0], receiver[1], receiver[2])

    # Mark the path in the matrix
    for point in complete_path:
        matrix[point[0], point[1], point[2]] = 1  # The order of indices is (x, y, z)

    return matrix


def simulate_ray_3d_with_reflection(nx, ny, nz, shooter, receiver):
    """
    Simulate the path of a ray shot from a shooter, reflected at the bottom layer, and received by a receiver.
    
    :param shooter: Tuple (x, y, z) for the shooter's position, z should be 0 for the top layer
    :param receiver: Tuple (x, y, z) for the receiver's position, z should be 0 for the top layer
    :param nz: The depth of the 3D volume, indicating the bottom layer where reflection occurs
    """

    matrix = np.zeros((nx, ny, nz))

    # Calculate the midpoint at the surface
    midpoint_x = (shooter[0] + receiver[0]) // 2
    midpoint_y = (shooter[1] + receiver[1]) // 2

    # The reflection point is directly below this midpoint at the bottom layer
    reflection_point = (midpoint_x, midpoint_y, nz-1)

    # Path from the shooter to the reflection point
    path_to_reflection = bresenham_line_3d(shooter[0], shooter[1], shooter[2], reflection_point[0], reflection_point[1], reflection_point[2])

    # Path from the reflection point to a point directly above the receiver
    path_from_reflection = bresenham_line_3d(reflection_point[0], reflection_point[1], reflection_point[2], receiver[0], receiver[1], receiver[2])

    # Combine the two paths for the complete ray trajectory
    complete_path = path_to_reflection[:-1] + path_from_reflection  # Exclude the reflection point from the first path to avoid duplication

    for point in complete_path:
        matrix[point[0], point[1], point[2]] = 1  # Note the order of indices (y, x) for numpy array

    return matrix


def ray_path_vertical(nx, ny, nd):
    G = np.zeros((nx, ny, nd*nd)) # For example, (64, 64, 32)
    start_point = np.array((0, 0))  # For example, (0, 0)
    end_point = np.array((0, ny-1))  # For example, (63, 63)
    idx = 0
    start_step, end_step = np.array((nx//nd, 0)), np.array((nx//nd, 0))
    while all(start_point < (nx, ny)):
        while all(end_point < (nx, ny)):
            # Draw & Collect
            G[:,:, idx] = simulate_ray_2d(nx, ny, start_point, end_point)
            # Update
            idx += 1
            end_point += end_step
            
        start_point += start_step
        end_point = np.array((0, ny-1))  # For example, (63, 63)
    return G


def ray_path_horizontal(nx, ny, nd):
    G = np.zeros((nx, ny, nd*nd)) # For example, (64, 64, 32)
    start_point = np.array((0, 0))  # For example, (0, 0)
    end_point = np.array((nx-1, 0))  # For example, (63, 63)
    idx = 0
    start_step, end_step = np.array((0, ny//nd)), np.array((0, ny//nd))
    while all(start_point < (nx, ny)):
        while all(end_point < (nx, ny)):
            # Draw & Collect
            G[:,:, idx] = simulate_ray_2d(nx, ny, start_point, end_point)
            # Update
            idx += 1
            end_point += end_step
            
        start_point += start_step
        end_point = np.array((nx-1, 0))  # For example, (63, 63)
    return G


def generate_ray_path(nx, ny, nd):
    Gv = ray_path_vertical(nx, ny, nd)
    Gh = ray_path_horizontal(nx, ny, nd)
    G = np.concatenate((Gv, Gh),axis=-1)
    G = G.reshape(G.shape[0]*G.shape[1], -1) # (nx*ny, nd)
    return G


def ray_path_cross_well_archive(nx, ny, nz, cross_center_x, cross_center_y, cross_radius, depths=None):

    ray_path = []

    depths = [0, 4, 9, 14, 19] if depths is None else depths

    receivers = [(cross_center_x, cross_center_y - cross_radius, i) for i in depths] 
    shooters = [(cross_center_x, cross_center_y + cross_radius, i) for i in depths]
    for shooter in shooters:
        for receiver in receivers:
            ray_path.append(simulate_ray_3d(nx, ny, nz, shooter, receiver)) 
    
    receivers = [(cross_center_y - cross_radius, cross_center_x, i) for i in depths] 
    shooters = [(cross_center_y + cross_radius, cross_center_x, i) for i in depths]
    for shooter in shooters:
        for receiver in receivers:
            ray_path.append(simulate_ray_3d(nx, ny, nz, shooter, receiver)) 

    return np.stack(ray_path, axis=-1)

 
def ray_path_cross_well(nx, ny, nz, cross_center_x, cross_center_y, cross_radius, 
                        depths=None, cross_angles=None):

    ray_path = []
    depths = [0, 4, 9, 14, 19] if depths is None else depths
    cross_angles = [0.0, np.pi/2., np.pi, np.pi*3/2] if cross_angles is None else cross_angles
    for angle in list(cross_angles):
        proj_ratio_x, proj_ratio_y = np.sin(angle), np.cos(angle)
        length_x = int(proj_ratio_x * cross_radius)
        length_y = int(proj_ratio_y * cross_radius)
        receivers = [(cross_center_x + length_x, cross_center_y + length_y, i) for i in depths] 
        shooters = [(cross_center_x, cross_center_y, i) for i in depths]
        for shooter in shooters:
            for receiver in receivers:
                ray_path.append(simulate_ray_3d(nx, ny, nz, shooter, receiver)) 

    return np.stack(ray_path, axis=-1)

