from sh import *
import torch
import open3d as o3d
import numpy as np
import statistics
import json


# LABELS AND THEIR COLORS
color_dict = {
    0 : torch.tensor([0.0, 0.0, 0.0]), # statue and sky
    1 : torch.tensor([35.0, 142.0, 107.0]), # color vegetation
    2 : torch.tensor([153.0, 153.0, 153.0]), # pole
    3 : torch.tensor([156.0, 102.0, 102.0]), # wall
    4 : torch.tensor([70.0, 70.0, 70.0]), # building
    5 : torch.tensor([255.0, 0.0, 0.0]), # vehicle
    6 : torch.tensor([232.0, 35.0, 244.0]), # sidewalk
    7 : torch.tensor([128.0, 64.0, 128.0]), # road
    8 : torch.tensor([50.0, 234.0, 157.0]), # roadline
    9 : torch.tensor([153.0, 153.0, 190.0]), # fence
}
color_lst = [
    torch.tensor([0.0, 0.0, 0.0]), # statue and sky
    torch.tensor([35.0, 142.0, 107.0]), # color vegetation
    torch.tensor([153.0, 153.0, 153.0]), # pole
    torch.tensor([156.0, 102.0, 102.0]), # wall
    torch.tensor([70.0, 70.0, 70.0]), # building
    torch.tensor([255.0, 0.0, 0.0]), # vehicle
    torch.tensor([232.0, 35.0, 244.0]), # sidewalk
    torch.tensor([128.0, 64.0, 128.0]), # road
    torch.tensor([50.0, 234.0, 157.0]), # roadline
    torch.tensor([153.0, 153.0, 190.0]), # fence
    torch.tensor([255.0, 255.0, 255.0]), # white
]
color_statue_and_sky = torch.tensor([0.0, 0.0, 0.0])
color_vegetation = torch.tensor([35.0, 142.0, 107.0])
color_pole = torch.tensor([153.0, 153.0, 153.0])
color_wall = torch.tensor([156.0, 102.0, 102.0])
color_building = torch.tensor([70.0, 70.0, 70.0])
color_vehicle = torch.tensor([255.0, 0.0, 0.0])
color_sidewalk = torch.tensor([232.0, 35.0, 244.0])
color_road = torch.tensor([128.0, 64.0, 128.0])
color_roadline = torch.tensor([50.0, 234.0, 157.0])
color_fence = torch.tensor([153.0, 153.0, 190.0])


class LabelCloud:

    def __init__(self, xyz, features, camera_poses, degree=3, colors_path=None, field_ids_path=None):

        self.xyz = torch.tensor(xyz, dtype=torch.float)
        self.colors = torch.zeros((self.xyz.shape), dtype=torch.float) # 0.0 - 1.0
        self.labels = torch.zeros((self.xyz.shape[0], 1), dtype=torch.int)
        if field_ids_path == None:
            self.field_ids = [[] for _ in range(xyz.shape[0])]
        else:
            with open(field_ids_path, 'r') as file:
                self.field_ids = json.load(file)

        if colors_path is None:
            num_of_splats = xyz.shape[0]
            num_of_cams = camera_poses.shape[0]

            for cam_idx in range(num_of_cams):
                if (cam_idx % 10 == 0):
                    print(round(cam_idx / num_of_cams * 100.0), "% done.")

                cam_pos = torch.tensor(camera_poses[cam_idx].reshape(1, -1))
                shs_view = features.transpose(1, 2).view(-1, 3, (degree + 1) ** 2)
                dir_pp = (self.xyz - cam_pos.repeat(features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(degree, shs_view, dir_pp_normalized)
                self.colors += torch.clamp_min(sh2rgb + 0.5, 0.0) / num_of_cams

            for splat_idx in range(num_of_splats):
                rgb = self.colors[splat_idx] * 255.0

                dist = torch.linalg.norm(rgb - color_statue_and_sky)
                label_rgb = color_statue_and_sky

                if torch.linalg.norm(rgb - color_vegetation) < dist:
                    label_rgb = color_vegetation
                    dist = torch.linalg.norm(rgb - color_vegetation)
                if torch.linalg.norm(rgb - color_pole) < dist:
                    label_rgb = color_pole
                    dist = torch.linalg.norm(rgb - color_pole)
                if torch.linalg.norm(rgb - color_wall) < dist:
                    label_rgb = color_wall
                    dist = torch.linalg.norm(rgb - color_wall)
                if torch.linalg.norm(rgb - color_building) < dist:
                    label_rgb = color_building
                    dist = torch.linalg.norm(rgb - color_building)
                if torch.linalg.norm(rgb - color_vehicle) < dist:
                    label_rgb = color_vehicle
                    dist = torch.linalg.norm(rgb - color_vehicle)
                if torch.linalg.norm(rgb - color_sidewalk) < dist:
                    label_rgb = color_sidewalk
                    dist = torch.linalg.norm(rgb - color_sidewalk)
                if torch.linalg.norm(rgb - color_road) < dist:
                    label_rgb = color_road
                    dist = torch.linalg.norm(rgb - color_road)
                if torch.linalg.norm(rgb - color_roadline) < dist:
                    label_rgb = color_roadline
                    dist = torch.linalg.norm(rgb - color_roadline)
                if torch.linalg.norm(rgb - color_fence) < dist:
                    label_rgb = color_fence
                    dist = torch.linalg.norm(rgb - color_fence)
                self.colors[splat_idx] = label_rgb / 255.0
        else:
            self.colors = torch.load(colors_path)
        self.update_labels_by_colors()

    def update_labels_by_colors(self):
        for i in range(self.colors.size()[0]):
            c = self.colors[i]*255.0
            j = 0
            equal = torch.all(torch.eq(c, color_lst[j])).item()
            while equal is False:
                j += 1
                equal = torch.all(torch.eq(c, color_lst[j])).item()
            self.labels[i] = j

    def update_colors_by_labels(self):
        for i in range(self.colors.size()[0]):
            self.colors[i] = color_lst[self.labels[i]] / 255.0

    def visualize(self, selected_label=None):
        if selected_label is not None:
            self.xyz = self.xyz[self.labels[:,0] != selected_label]
            self.colors = self.colors[self.labels[:,0] != selected_label]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        o3d.visualization.draw_geometries([pcd])

    def save_colors(self, file_path):
        torch.save(self.colors, file_path)

    def show_field(self, field_id):
        pcl = np.array([[0.0, 0.0, 0.0]])
        color = np.array([[0.0, 0.0, 0.0]])
        for i in range(self.xyz.shape[0]):
            if field_id in self.field_ids[i]:
                pcl = np.append(pcl, self.xyz[i].reshape(1, -1), axis=0)
                color = np.append(color, self.colors[i].reshape(1, -1), axis=0)


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd])

    '''def get_points_labels_and_indices(self, field_id):
        coord = np.array([[0.0, 0.0, 0.0]])
        coord_t = torch.zeros(0, 0)
        color = np.array([[0.0, 0.0, 0.0]])
        index =
        for i in range(self.xyz.shape[0]):
            if field_id in self.field_ids[i]:
                coord = np.append(pcl, self.xyz[i].reshape(1, -1), axis=0)
                color = np.append(color, self.colors[i].reshape(1, -1), axis=0)'''


    def smoothing(self, res_x, res_z, overlap=1.0, radius=1.0, minimum_neighbor=2):
        #self.set_fields(res_x, res_z, overlap)
        field_number = res_x * res_z

        for f_id in range(field_number):
            if f_id > 600:
                print(f_id, " / ", field_number, "fields")

                counter = 0
                for counter_i in range(len(self.field_ids)):
                    if f_id in (self.field_ids[counter_i]):
                        counter += 1
                print("Number of points: ", counter)

                for l_idx in range(len(self.labels)):
                    if f_id in (self.field_ids[l_idx]):

                        label_lst = []

                        for n_idx in range(len(self.labels)):
                            if f_id in (self.field_ids[n_idx]):

                                dist = torch.linalg.norm(self.xyz[l_idx] - self.xyz[n_idx])
                                if dist < radius:
                                    label_lst.append(self.labels[n_idx].item())

                        if len(label_lst) < minimum_neighbor:
                            self.labels[l_idx] = 10
                        else:
                            self.labels[l_idx] = statistics.mode(label_lst)

        self.update_colors_by_labels()
        self.save_colors("data\color_file_smooth_rad_1_all_with_whites_NEW.pt")

    def get_color(self):
        print("get_color")
        print(self.colors.size())
        color_rgbs  = self.colors.unsqueeze(1)
        print(color_rgbs.size())
        color_dc = RGB2SH(color_rgbs)
        print(color_dc.size())
        return color_dc

    def denoising(self, radius=2, minimum_neighbors=5):
        pcl = gs.xyz[0].reshape(1, -1)
        for i in range(gs.size):
            if i % 10 == 0: print(round(i / gs.size*100.0), "% done.")
            actual = gs.xyz[i]
            neighbors = gs.get_closest_points_indices(actual, 1.0)
            if len(neighbors) > 3:
                pcl = np.append(pcl, actual.reshape(1, -1), axis=0)
        return True

    def set_fields(self, res_x, res_z, overlap):
        res_board = torch.arange(0, res_x * res_z, 1).reshape(res_x, res_z)
        print(res_board)

        min_x = -110.0
        max_x = 120.0
        min_z = -80.0
        max_z = 150.0

        bounds_x = []
        bounds_z = []

        range_x = abs(min_x) + abs(max_x)
        range_z = abs(min_z) + abs(max_z)

        step_x = range_x / res_x
        step_z = range_z / res_z

        set_x_bounds = min_x + step_x
        while (set_x_bounds < max_x):
            bounds_x.append(set_x_bounds)
            set_x_bounds += step_x
        bounds_x.append(max_x)

        set_z_bounds = min_z + step_z
        while (set_z_bounds < max_z):
            bounds_z.append(set_z_bounds)
            set_z_bounds += step_z
        bounds_z.append(max_z)

        # print(bounds_x)
        # print(bounds_z)

        pcl = np.array([[0.0, 0.0, 0.0]])
        for i in range(self.xyz.shape[0]):
            if (i % 2000 == 0):
                print("Setting fields: ", round(i / self.xyz.shape[0] * 100.0), "% done.")
            field_ids = []

            x_fields = []
            i_x = 0
            # lower
            while i_x < len(bounds_x) - 1 and (self.xyz[i][0] > (bounds_x[i_x]) - overlap):
                i_x += 1
            x_fields.append(i_x)

            # upper
            if i_x > 0 and self.xyz[i][0] < bounds_x[i_x - 1] + overlap:
                x_fields.append(i_x - 1)

            z_fields = []
            i_z = 0
            # lower
            while i_z < len(bounds_z) - 1 and (self.xyz[i][2] > (bounds_z[i_z]) - overlap):
                i_z += 1
            z_fields.append(i_z)
            # upper
            if i_z > 0 and self.xyz[i][2] < bounds_z[i_z - 1] + overlap:
                z_fields.append(i_z - 1)

            for xf in x_fields:
                for zf in z_fields:
                    field_ids.append(res_board[xf, zf].item())

            self.field_ids[i] = field_ids

    def get_closest_points_indices(self, point, radius=1.0):
        indices = []
        for i in range(0, self.xyz.shape[0]):
            actual = self.xyz[i]
            d = np.linalg.norm(actual - point)
            if d < radius:
                indices.append(i)
        return indices