import numpy as np
import cv2

class HomographyTransformer:
    def __init__(self, pts_image_plane, pts_ground_plane, scale_to_m = 1.0):
        if len(pts_image_plane) != len(pts_ground_plane):
            raise RuntimeError("ERROR: PTS_GROUND_PLANE and PTS_IMAGE_PLANE should be of same length")

        np_pts_ground = np.array(pts_ground_plane)
        np_pts_ground = np_pts_ground * scale_to_m
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(pts_image_plane)
        np_pts_image = np_pts_image * scale_to_m
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)
        self.h_inv = np.linalg.inv(self.h)


    def transform_uv_to_xy(self, point):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns a normal non-np 1x2 matrix of xy displacement vector from the
        camera to the point on the ground plane.
        Camera points along positive x axis and y axis increases to the left of
        the camera.

        Units are in meters.
        """
        u, v = point
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(self.h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return np.array([x, y], float)

    def transform_xy_to_uv(self, point):
        x, y = point
        homogeneous_point = np.array([[x], [y], [1]])
        uv = np.dot(self.h_inv, homogeneous_point)
        scaling_factor = 1.0 / uv[2, 0]
        homogeneous_uv = uv * scaling_factor
        u = homogeneous_uv[0, 0]
        v = homogeneous_uv[1, 0]
        return np.array([u, v], int)