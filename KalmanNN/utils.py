import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

def _xyz(key:str='') -> list:
    """Returns ['key_x', 'key_y', 'key_z']"""
    return [f'{key}_{xyz}' for xyz in ['x', 'y', 'z']]


def orthogonal(k: np.array, x: np.array = None) -> np.array:
    """Returns vector, orthogonal to the given k, that lies in the same
    plane as x and k vectors

    Note:
        If x is not given, it is generated randomly
    """
    # Take a random vector if x is not given
    if x is None:
        x = np.random.randn(3)                    # take a random vector

    # Convert array-like objects to arrays
    x = np.array(x, dtype=np.float64)
    k = np.array(k, dtype=np.float64)

    x -= x.dot(k) * k / np.linalg.norm(k)**2      # make it orthogonal to k
    x = normalize(x)
    return x


def normalize(v:np.array) -> np.array:
    """would like to convert a NumPy array to a unit vecto"""
    if np.isinf(v).any():
        return 0
    else:
        return v / (np.linalg.norm(v) + 1e-20)  # 1e20 is div by 0 protection


def get_transformation_matrix(p, p_prime):
    '''
    Find the unique homogeneous affine transformation that
    maps a set of 3 points to another set of 3 points in 3D
    space:

        p_prime == np.dot(p, R) + t

    where `R` is an unknown rotation matrix, `t` is an unknown
    translation vector, and `p` and `p_prime` are the original
    and transformed set of points stored as row vectors:

        p       = np.array((p1,       p2,       p3))
        p_prime = np.array((p1_prime, p2_prime, p3_prime))

    The result of this function is an augmented 4-by-4
    matrix `A` that represents this affine transformation:

        np.column_stack((p_prime, (1, 1, 1))) == \
            np.dot(np.column_stack((p, (1, 1, 1))), A)

    Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q       = p[1:]       - p[0]
    Q_prime = p_prime[1:] - p_prime[0]

    # calculate rotation matrix
    R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
               np.row_stack((Q_prime, np.cross(*Q_prime))))

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix
    return np.column_stack((np.row_stack((R, t)),
                            (0, 0, 0, 1)))


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    See https://stackoverflow.com/a/50664367

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    def set_axes_radius(ax, origin, radius):
        '''See https://stackoverflow.com/a/50664367'''
        ax.set_xlim([origin[0] - radius, origin[0] + radius])
        ax.set_ylim([origin[1] - radius, origin[1] + radius])
        ax.set_zlim([origin[2] - radius, origin[2] + radius])

    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def plot(data, title=''):
    # display some trajectories for visual representation
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    data = np.array(data)
    colors = np.arange(data.shape[0])

    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap='plasma');
    ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))  # aspect ratio is 1:1:1 in data space
    set_axes_equal(ax)
    plt.show()

def distance_metrics(preds, gt):
    '''Compute MAD, FAD, AvgScore and WindowErr distance metrics.

    Both `gt` and `preds` are expected to be torch.tensors of shape
    [batch, horizon, variables]

    MAD: Mean Avg Distance (mean of L-2 norm between traj. points)
    FAD: Final Avg Distance (L-2 norm between two last traj. points)
    AvgScore: MAD * FAD
    WindowErr: MAD for each corresponding pair of traj. points
    '''
    errors = torch.norm((gt - preds), 2, -1).mean(0)
    mad, fad = errors.mean(), errors[-1]
    return mad, fad, mad*fad, errors
