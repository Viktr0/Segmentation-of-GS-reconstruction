import torch
import open3d as o3d
import numpy as np
import statistics
import json
from math import sqrt, acos
import cv2


class Camera:
    def __init__(self, center, direction, name):
        self.pos = center

        # a and b parameters are calculated from the camera intrinsics.
        a = 2 * sqrt(3)
        b = 1.948557159

        self.corners = np.array([[a/2, b/2, 1.0], # Tl -> BR
                                [a/2, -b/2, 1.0], # BL -> TR
                                [-a/2, b/2, 1.0], # TR -> BL
                                [-a/2, -b/2, 1.0]]) # BR -> TL
        self.set_corners(direction) # 0-TopLeft, 1-BottomLeft, 2-TopRight, 3-BottomRight

        self.image = ImagePixels(name + ".png")

    def describe(self):
        print(self.pos)
        print(self.corners)

    def axis_with_angles(self, direction):
        '''
        Calculate axis with angles from quaternion vector [w,x,y,z]. Returns with (Roll, Pitch, Yaw) in this order.
        '''
        angle = 2 * acos(direction[0])
        axis_0 = direction[1] * angle
        axis_1 = direction[2] * angle
        axis_2 = direction[3] * angle
        return axis_0, axis_1, axis_2

    def rotate(self, pitch, roll, yaw):
        '''
        Pitch is around Y axis. Yaw is around Z axis. Roll is around X axis.
        '''
        cosa = np.cos(yaw)
        sina = np.sin(yaw)

        cosb = np.cos(pitch)
        sinb = np.sin(pitch)

        cosc = np.cos(roll)
        sinc = np.sin(roll)

        Axx = cosa * cosb
        Axy = cosa * sinb * sinc - sina * cosc
        Axz = cosa * sinb * cosc + sina * sinc

        Ayx = sina * cosb
        Ayy = sina * sinb * sinc + cosa * cosc
        Ayz = sina * sinb * cosc - cosa * sinc

        Azx = -sinb
        Azy = cosb * sinc
        Azz = cosb * cosc

        for c in range(len(self.corners)):
            px = self.corners[c][0]  # x-coordinate of the point
            py = self.corners[c][1]  # y-coordinate of the point
            pz = self.corners[c][2]  # z-coordinate of the point

            self.corners[c][0] = Axx * px + Axy * py + Axz * pz
            self.corners[c][1] = Ayx * px + Ayy * py + Ayz * pz
            self.corners[c][2] = Azx * px + Azy * py + Azz * pz

    def translate(self, translation_vec):
        '''
        Translate the camera to it's position in world coordinate system.
        :param translation_vec: The position of the camera
        '''
        for c in range(len(self.corners)):
            self.corners[c] = self.corners[c] + translation_vec

    def set_corners(self, direction):
        '''
        Set the coorners to the corresponding camera data.
        '''
        r, p, y = self.axis_with_angles(direction)
        self.rotate(p, r, y)
        self.translate(self.pos)

    def get_ray_by_pixel(self, i, j):
        '''
        Returns the ray that starts from the position of the camera and goes to the (i,j) pixel.
        :param i: row
        :param j: columns
        '''
        right = (self.corners[1] - self.corners[3]) / 1280
        down = (self.corners[2] - self.corners[3]) / 720

        pixel_coord = self.corners[3] + i * down + j * right

        dir = pixel_coord - self.pos
        unit_dir = dir / np.linalg.norm(dir)

        ray = Ray(self.pos, unit_dir)
        return ray


class Ray:
    def __init__(self, start, direction):
        self.start = start
        self.dir = direction

    def get_points(self, length=10, number_of_points = 100):
        '''
        For visualization purposes.
        '''
        ray_pcl = self.start.reshape(1,-1)
        for i in range(1, number_of_points):
            new_p = self.start + length/number_of_points * i * self.dir.reshape(1,-1)
            ray_pcl = np.append(ray_pcl, new_p, axis=0)
        return ray_pcl


class ImagePixels:
    def __init__(self, name):
        file_path = "data\images" + "\\" + name
        self.image = cv2.imread(file_path)
        #self.mask = np.ones((self.image.shape[0], self.image.shape[1]))

    def getUniqueColors(self):

        rows, cols, _ = self.image.shape

        unique_colors = []
        for i in range(rows):
            for j in range(cols):
                c = self.image[i, j]
                color = [c[0], c[1], c[2]]
                if self.isUnique(color, unique_colors):
                    unique_colors.append([color[0], color[1], color[2]])
        return unique_colors

    def isUnique(self, color, colors):

        if len(colors) == 0:
            return True
        else:
            for c in colors:
                if (c[0] == color[0]) and (c[1] == color[1]) and (c[2] == color[2]):
                    return False
            return True

    def create_mask(self, colors):

        self.mask = np.zeros((self.image.shape[0], self.image.shape[1]))
        colors = colors.reshape(-1,3)
        for i in range(0, colors.shape[0]):
            self.mask = self.mask + cv2.inRange(self.image, colors[i], colors[i])

    def show_image(self, is_masked=False):

        if is_masked:
            self.image[self.mask == 0] = np.array([0,0,0]) #np.array([255, 255, 255])

        cv2.imshow("Loaded Image", self.image)
        cv2.waitKey(0)
        # closing all open windows
        cv2.destroyAllWindows()

    def count_masked_pixels(self):
        rows, cols, _ = self.image.shape
        count = 0
        for i in range(rows):
            for j in range(cols):
                if self.mask[i,j] != 0:
                    count += 1
        return count

    def set_labels(self):
        rows, cols, _ = self.image.shape
        for i in range(rows):
            for j in range(cols):
                c = self.image[i, j]
                self.labels[i, j] = assign_label(c)


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