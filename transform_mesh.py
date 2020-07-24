import cv2
import json
import numpy as np
from matplotlib import pyplot


image_file = "/home/robin/Desktop/sample/helen-trainset-1010057391_1.jpg"
ibug_file = "/home/robin/Desktop/sample/helen-trainset-1010057391_1.json"
mesh_file = "/home/robin/Desktop/sample/helen-trainset-1010057391_1_m468.json"

epn_width = 20


def get_distance(point1, point2):
    """Calculate the distance between two points."""
    return np.linalg.norm(point2 - point1)


def get_angle(vector2, vector1):
    """Return the angel between two vectors."""
    d = np.dot(vector1, vector2)
    cos_angle = d / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    if cos_angle > 1.0:
        angle = 0
    elif cos_angle < -1.0:
        angle = np.pi
    else:
        angle = np.arccos(cos_angle)

    return angle


def rotate(points, radius, center):
    """Rotate the points by angle"""
    _points = points - np.array(center, np.float)
    cos_angle = np.cos(radius)
    sin_angle = np.sin(radius)
    rotaion_matrix = np.array([[cos_angle, sin_angle],
                               [-sin_angle, cos_angle]])

    return np.dot(_points, rotaion_matrix) + center


if __name__ == "__main__":
    # Read in the image.
    image = pyplot.imread(image_file)

    # Read in the IBUG marks.
    with open(ibug_file, 'r') as f:
        data = json.load(f)
        ibug_points = np.reshape(data, (-1, 2)) * 128

    # Read in the MESH points.
    with open(mesh_file, 'r') as f:
        data = json.load(f)
        mesh_points = np.reshape(data, (-1, 3)) * (128 + epn_width * 2)

    # The IBUG data are manually annotated, which I believe could be used to
    # calibrate the mesh points. Here are some points of interest that is more
    # stable than others. They are:
    # * left eye left corner
    # * left eye right corner
    # * right eye left corner
    # * right eye right corner
    # * mouse left corner
    # * mouse right corner

    anchor_points_ibug = np.array([ibug_points[36],
                                   ibug_points[39],
                                   ibug_points[42],
                                   ibug_points[45],
                                   ibug_points[48],
                                   ibug_points[54]], np.float32)
    anchor_points_mesh = np.array([mesh_points[33],
                                   mesh_points[133],
                                   mesh_points[362],
                                   mesh_points[263],
                                   mesh_points[78],
                                   mesh_points[308]], np.float32)

    # The mesh are of 3D dimensions.
    anchor_points_mesh = anchor_points_mesh[:, :2]
    mesh_to_transform = mesh_points[:, :2]

    # TODO: transform the mesh points.
    # 1. scale
    scale = 0.5
    mesh_transformed = mesh_to_transform * scale

    # 2. translation
    # 3. rotation

    # Draw IBUG marks.
    fig_ibug = pyplot.figure()
    ibug_plot = fig_ibug.add_subplot(111)
    ibug_plot.imshow(image)
    ibug_lines = ibug_plot.plot(ibug_points[:, 0], ibug_points[:, 1],
                                color='yellow', marker='.',
                                linestyle='None', markersize=6)
    ibug_plot.plot(anchor_points_ibug[:, 0], anchor_points_ibug[:, 1],
                   color='blue', marker='.',
                   linestyle='None', markersize=6)

    # Draw mesh marks.
    fig_mesh = pyplot.figure()
    mesh_plot = fig_mesh.add_subplot(111)
    image = cv2.copyMakeBorder(image,
                               epn_width, epn_width, epn_width, epn_width,
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
    mesh_plot.imshow(image)
    mesh_lines = mesh_plot.plot(mesh_points[:, 0], mesh_points[:, 1],
                                color='yellow', marker='.',
                                linestyle='None', markersize=3)
    mesh_plot.plot(anchor_points_mesh[:, 0], anchor_points_mesh[:, 1],
                   color='blue', marker='.',
                   linestyle='None', markersize=6)

    # Draw transformed mesh.
    fig_mesh = pyplot.figure()
    t_mesh_plot = fig_mesh.add_subplot(111)
    t_mesh_plot.imshow(image)
    t_mesh_lines = t_mesh_plot.plot(mesh_transformed[:, 0], mesh_transformed[:, 1],
                                    color='yellow', marker='.',
                                    linestyle='None', markersize=3)
    pyplot.show()
