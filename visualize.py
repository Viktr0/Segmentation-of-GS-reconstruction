from ellipsoid_list import EllipsoidList
from camera_list import CameraList
from math import sqrt
import torch
import open3d as o3d
from ellipsoid import Ellipsoid
import statistics
import json
import numpy as np


color_lst = [
    torch.tensor([255.0, 255.0, 255.0]), # white
    torch.tensor([35.0, 142.0, 107.0]), # color vegetation
    torch.tensor([0.0, 0.0, 0.0]), # statue and sky
    torch.tensor([153.0, 153.0, 153.0]), # pole
    torch.tensor([156.0, 102.0, 102.0]), # wall
    torch.tensor([70.0, 70.0, 70.0]), # building
    torch.tensor([255.0, 0.0, 0.0]), # vehicle
    torch.tensor([232.0, 35.0, 244.0]), # sidewalk
    torch.tensor([128.0, 64.0, 128.0]), # road
    torch.tensor([50.0, 234.0, 157.0]), # roadline
    torch.tensor([153.0, 153.0, 190.0]), # fence
]


class Visualizer:
    def __init__(self, ellipsoid_list):
        self.el = ellipsoid_list

    def visualize(self, load_path, segment_list=[], show_ellipsoids=False):

        with open(load_path, 'r') as file:
            label_lists = json.load(file)

        if len(segment_list) == 0:
            segment_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        counter = 0
        coords = np.zeros((1, 3))
        colors = np.zeros((1, 3))
        for i in range(0, len(label_lists)):
            if len(label_lists[i]) != 0:
                counter += 1
                modus = statistics.mode(label_lists[i])
                if modus in segment_list:
                    c = self.el.ellipsoids[i].center
                    coords = np.append(coords, c.reshape(1, -1), axis=0)
                    colors = np.append(colors, (color_lst[modus] / 255.0).reshape(1, -1), axis=0)
                    if show_ellipsoids:
                        c_m = self.el.ellispoids[i].cov_mat
                        new_ellipsoid = Ellipsoid(c, c_m)
                        coords = np.append(coords, new_ellipsoid.get_points(steps=10), axis=0)
        print(counter)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        if show_ellipsoids is False:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':

    el_path = "data\point_cloud.ply"
    el = EllipsoidList(polygon_data_path=el_path)

    visualizer = Visualizer(ellipsoid_list=el)

    label_path = "data\labels_0-245.json" # "data\labels_100.json"
    s_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Visualization of the labeled ellipsoids by their centers
    visualizer.visualize(load_path=label_path, segment_list=s_list, show_ellipsoids=False)