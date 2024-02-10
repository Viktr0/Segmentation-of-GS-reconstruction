from math import exp
from typing import List, Any
from covariance_operations import *
import plyfile
from ellipsoid import *
import torch
import numpy as np
import cv2


class EllipsoidList:
    def __init__(self, polygon_data_path=None, ellipsoid_list=None):

        if ellipsoid_list is None:
            # Load json file
            plydata = plyfile.PlyData.read(polygon_data_path)
            print(f"PLY data length: \t {plydata.elements[0]['x'].size}")

            # XYZ coordinates
            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])), axis=1)

            # Number of ellipsoids
            self.size = xyz.shape[0]

            # Opacity
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
            opacities = set_opacities(opacities)

            # Scale
            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

            # Rotations
            rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

            # Covariance matrices
            cov_mats = get_covariance(torch.tensor(scales), torch.tensor(rots))

            # Create ellipsoid list with unique IDs
            self.ellipsoids = []
            for i in range(0, self.size):
                self.ellipsoids.append(Ellipsoid(xyz[i], cov_mats[i].to('cpu').numpy(), opacities[i], i))

        else:
            self.size = len(ellipsoid_list)
            self.ellipsoids = ellipsoid_list

    def reset_indexes(self):
        index = 0
        for e in self.ellipsoids:
            e.id = index
            index += 1

    def get_centers(self):
        centers = []
        for e in self.ellipsoids:
            centers.append(e.center)
        return centers

    def get_mean_point(self):

        points_sum = np.array([0.0, 0.0, 0.0])

        for e in self.ellipsoids:
            points_sum += e.center

        mean = points_sum / self.size
        return mean

    def get_min_max_of_coord(self, coordinate):

        minimum = 1000.0
        maximum = -1000.0

        for e in self.ellipsoids:
            if minimum > e.pos[coordinate]:
                minimum = e.pos[coordinate]
            if maximum < e.pos[coordinate]:
                maximum = e.pos[coordinate]

        print(f"{coordinate}. coordinate's range: {minimum} - {maximum}")
        return minimum, maximum

    def get_closest_ellipsoids(self, point, radius=1.0):

        closest_ellipsoids = []

        for e in self.ellipsoids:
            d = np.linalg.norm(e.center - point)
            if d < radius:
                closest_ellipsoids.append(e)

        return closest_ellipsoids

    def get_ellipsoids_above_opacity_lvl(self, opacity_lvl):

        ellipsoids_above_op_lvl = []

        for e in self.ellipsoids:
            if e.op > opacity_lvl:
                ellipsoids_above_op_lvl.append(e)

        return ellipsoids_above_op_lvl

    def get_ellipsoids_around_ray(self, ray, radius=5.0):

        ellipsoids_around_ray = []

        for e in self.ellipsoids:
            if is_point_around_ray(e.center, ray, radius):
                ellipsoids_around_ray.append(e)

        return ellipsoids_around_ray


def set_opacities(opacities):
    size = len(opacities)
    for i in range(0, size):
        opacities[i] = sigmoid(opacities[i])
    return opacities


def sigmoid(x):
    return 1 / (1 + exp(-x))


def is_point_around_ray(point, ray, radius):
    # Calculate vector from C (cylinder center) to P
    CP = point - ray.start

    # Calculate the projection of CP onto the cylinder axis direction
    projection_length = np.dot(CP, ray.dir)
    if projection_length < 0:
        return False
    projection = ray.start + projection_length * ray.dir

    # Calculate the distance from the projection to P
    distance = np.linalg.norm(point - projection)

    # Check if the distance is less than or equal to the cylinder's radius
    return distance <= radius


def get_distance(p1, p2):
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def get_closest_index(x, y, z, pc):
    p1 = np.array([x, y, z])
    dist = 100
    closest_index = -1
    r = np.asarray(pc.points).shape[0]
    for i in range(r):
        p2 = np.asarray(pc.points)[i]
        ad = get_distance(p1, p2)
        if ad < dist:
            dist = ad
            closest_index = i
    return closest_index


def get_near_points_indexes(p1, radius, pc):
    points = np.asarray(pc.points)
    idxes: List[Any] = []
    for i in range(points.shape[0]):
        p2 = points[i]
        distance = get_distance(p1, p2)
        if distance < radius:
            idxes.append(i)
    return idxes


def draw_circle(x, y, z, pc, r, g, b, radius, center_dot=False):
    idx = get_closest_index(x, y, z, pc)
    center = np.asarray(pc.points)[idx]
    indexes = get_near_points_indexes(center, radius, pc)
    for i in indexes:
        np.asarray(pc.colors)[i] = [r, g, b]
    if center_dot:
        np.asarray(pc.colors)[idx] = [0, 1, 0]