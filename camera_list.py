import math
from camera import Camera
import numpy as np
import json


class CameraList:
    def __init__(self, camera_data_path):

        with open(camera_data_path, "r") as f:
            camera_data_json = json.load(f)

        self.size = len(camera_data_json)
        print(f"Camera data length: ", self.size)

        #self.width = 1280
        #self.height = 720
        #self.focal = 369.5041722813606
        #self.fovX = self.focal_to_fov(self.focal, self.width)
        #self.fovY = self.focal_to_fov(self.focal, self.height)

        positions = self.load_from_json(camera_data_json, "position")
        directions = self.directions_from_rotmats(camera_data_json)
        image_name = self.load_from_json(camera_data_json, "img_name")

        self.cameras = []
        for i in range(0, self.size):
            self.cameras.append(Camera(positions[i], directions[i], image_name[i]))

    def get_centers(self):
        centers = []
        for c in self.cameras:
            centers.append(c.pos)

        return centers

    def load_from_json(self, camera_data_json, property):

        list = []
        size = len(camera_data_json)
        for i in range(0, size):
            list.append(camera_data_json[i][property])
        return np.array(list)

    def directions_from_rotmats(self, camera_data_json):

        mats = self.load_from_json(camera_data_json, "rotation")
        size = mats.shape[0]
        dirs = self.rotmat_to_qvec(mats[0]).reshape(1, -1)
        for i in range(1, size):
            new_dir = self.rotmat_to_qvec(mats[i]).reshape(1, -1)
            dirs = np.append(dirs, new_dir, axis=0)
        return dirs

    def rotmat_to_qvec(self, R : np.array):

        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flatten()
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec

    def focal_to_fov(self, focal, pixels):

        return 2 * math.atan(pixels / (2 * focal))