import numpy as np
from camera import *
import torch
import open3d as o3d
import numpy as np
import statistics
import json


class Ellipsoid:
    def __init__(self, center_point, covariance_mat, opacity, index=-1):

        self.center = center_point
        self.cov_mat = covariance_mat
        self.op = opacity
        self.id = index

    def describe(self):
        print(self.center)
        print(self.cov_mat)
        print(self.op)
        print(self.id)

    def isOnSurface(self, point):

        diff = np.array(point - self.center)
        xM = np.dot(diff, np.linalg.inv(self.cov_mat))
        xMx = np.dot(xM, diff.reshape(-1, 1))

        return xMx > 0.999 and xMx < 1.001

    def get_points(self, steps=100):

        points = [[self.center[0], self.center[1], self.center[2]]]
        theta = np.linspace(0, 2 * np.pi, steps)
        phi = np.linspace(0, np.pi, steps)

        for t in theta:
            for p in phi:
                x = 100.0 * np.cos(t) * np.sin(p)
                y = 100.0 * np.sin(t) * np.sin(p)
                z = 100.0 * np.cos(p)

                sphere_point = self.center + np.array([x, y, z])
                dir = self.center - sphere_point
                unit_dir = dir / np.linalg.norm(dir)
                ray = Ray(sphere_point, unit_dir)
                d = self.compute_intersections(ray)
                if d != 1000000:
                    #print(d)
                    intersection = sphere_point + d * unit_dir
                    points.append([intersection[0], intersection[1], intersection[2]])

        return np.array(points)

    def compute_intersections(self, ray):

        dx = ray.dir[0]
        dy = ray.dir[1]
        dz = ray.dir[2]

        cx = self.center[0] - ray.start[0]
        cy = self.center[1] - ray.start[1]
        cz = self.center[2] - ray.start[2]

        A = np.linalg.inv(self.cov_mat)

        a = A[0, 0] * dx * dx + A[1, 1] * dy * dy + A[2, 2] * dz * dz + (A[1, 0] + A[0, 1]) * dx * dy + (
                A[2, 0] + A[0, 2]) * dx * dz + (A[2, 1] + A[1, 2]) * dy * dz
        b = -1 * (2 * A[0, 0] * dx * cx + (A[1, 0] + A[0, 1]) * dx * cy + (A[2, 0] + A[0, 2]) * dx * cz
                  + (A[1, 0] + A[0, 1]) * dy * cx + 2 * A[1, 1] * dy * cy + (A[2, 1] + A[1, 2]) * dy * cz
                  + (A[2, 0] + A[0, 2]) * dz * cx + (A[2, 1] + A[1, 2]) * dz * cy + 2 * A[2, 2] * dz * cz)
        c = (A[0, 0] * cx + A[1, 0] * cy + A[2, 0] * cz) * cx + (A[0, 1] * cx + A[1, 1] * cy + A[2, 1] * cz) * cy + (
                A[0, 2] * cx + A[1, 2] * cy + A[2, 2] * cz) * cz - 1

        if (b * b - 4 * a * c < 0):
            return 1000000

        t1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        t2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)

        if t2 < 0 and t1 < 0:
            return 1000000

        elif t2 < 0 and t1 >= 0:
            return t1

        elif t2 >= 0 and t1 < 0:
            return t2

        elif t2 < t1:
            return t2

        else:
            return t1