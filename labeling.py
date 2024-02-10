from ellipsoid_list import EllipsoidList
from camera_list import CameraList
from math import sqrt
import torch
import open3d as o3d
from ellipsoid import Ellipsoid
import statistics
import json
import numpy as np


def assign_label(color):
    if np.sum(np.array([107, 142, 35])) == np.sum(color): return 1      # color_vegetation
    elif np.sum(np.array([0, 0, 0])) == np.sum(color): return 2         # color_statue_and_sky
    elif np.sum(np.array([153, 153, 153])) == np.sum(color): return 3   # color_pole
    elif np.sum(np.array([102, 102, 156])) == np.sum(color): return 4   # color_wall
    elif np.sum(np.array([70, 70, 70])) == np.sum(color): return 5      # color_building
    elif np.sum(np.array([0, 0, 255])) == np.sum(color): return 6       # color_vehicle
    elif np.sum(np.array([244, 35, 232])) == np.sum(color): return 7    # color_sidewalk
    elif np.sum(np.array([128, 64, 128])) == np.sum(color): return 8    # color_road
    elif np.sum(np.array([157, 234, 50])) == np.sum(color): return 9    # color_roadline
    elif np.sum(np.array([190, 153, 153])) == np.sum(color): return 10  # color_fence
    else:
        return -1


def labeling(el, cl, camera_id_range, save_path, resolution_divider=100, opacity_threshold=0.9, load_path=""):

    if load_path == "":
        label_lists = [[] for _ in range(el.size)]
    else:
        with open(load_path, 'r') as file:
            label_lists = json.load(file)

    divider = int(sqrt(resolution_divider))
    num_of_pixels = (int(720/divider)+1) * (int(1280/divider)+1)
    num_of_cameras = len(camera_id_range)

    # OPACITY FILTER
    eo = EllipsoidList(ellipsoid_list=el.get_ellipsoids_above_opacity_lvl(opacity_threshold))

    camera_counter = 1
    for camera_idx in camera_id_range:
        cam = cl.cameras[camera_idx]

        camera_ray = cam.get_ray_by_pixel(720 / 2, 1280 / 2)
        es = EllipsoidList(ellipsoid_list=eo.get_ellipsoids_around_ray(camera_ray, radius=100.0))

        pixel_counter = 1
        for i in range(0, 720):
            for j in range(0, 1280):

                if i % divider == 0 and j % divider == 0:

                    print(f"Camera: {camera_counter}/{num_of_cameras} \t Ray: {pixel_counter}/{num_of_pixels}")

                    label = assign_label(cam.image.image[i,j])
                    ray = cam.get_ray_by_pixel(i, j)

                    er = EllipsoidList(ellipsoid_list=es.get_ellipsoids_around_ray(ray, radius=1.5))

                    distance = 1000000
                    intersected_splat_idx = -1
                    for e in er.ellipsoids:

                        d = e.compute_intersections(ray)
                        if d < distance:
                            distance = d
                            intersected_splat_idx = e.id

                    if distance < 1000000:
                        intersection = ray.start + distance * ray.dir
                        label_lists[intersected_splat_idx].append(label)
                    pixel_counter += 1
        camera_counter += 1

    with open(save_path, 'w') as file:
        json.dump(label_lists, file)


if __name__ == '__main__':

    el_path = "data\point_cloud.ply"
    el = EllipsoidList(polygon_data_path=el_path)

    cl_path = "data\cameras.json"
    cl = CameraList(camera_data_path=cl_path)

    save_path = "data\labels_0-545.json" # "data\labels_0-4_and_100.json"
    load_path = "data\labels_0-245.json" # "data\labels_100.json"
    cd_index_range = range(245, 545, 10) # range(5, 1486, 10)
    res_div = 100

    # Assign labels to ellipsoids by a defined range of images
    labeling(el, cl, cd_index_range, save_path, resolution_divider=res_div, load_path=load_path)