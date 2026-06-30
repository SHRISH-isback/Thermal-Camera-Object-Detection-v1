import cv2
import numpy as np
import config


class SpatialTransformer:
    """
    Trigonometric & Coordinate Engine.

    Step 8 optimization: the legacy pipeline called cv2.undistort + cv2.warpPerspective on
    the entire dense image every frame, just to later read out a handful of contour
    centroids. This version undistorts/rectifies ONLY the extracted 2D centroid
    coordinates -- a single matrix multiply per target instead of a full-frame remap --
    then projects them onto the physical ground plane via the pinhole/FOV trigonometry.
    """

    def __init__(self):
        self.K = config.CAMERA_MATRIX_NORM
        self.K_inv = np.linalg.inv(self.K)
        self.dist = config.DIST_COEFFS
        self.recalibrate_geometry()

    def recalibrate_geometry(self):
        fov_x_rad = np.deg2rad(config.SENSOR_FOV_DEGREES)
        fov_y_rad = np.deg2rad(config.SENSOR_FOV_DEGREES_V)

        # Total ground footprint covered by the camera's field of view, in inches
        self.view_width_inches = 2 * config.CAMERA_HEIGHT_CM * np.tan(fov_x_rad / 2) * 0.393701
        self.view_height_inches = 2 * config.CAMERA_HEIGHT_CM * np.tan(fov_y_rad / 2) * 0.393701

        self.fov_x_rad = fov_x_rad
        self.fov_y_rad = fov_y_rad

        # Retained for any dashboard-side pixel<->inch scaling needs
        self.inches_per_pixel = self.view_width_inches / config.UPSCALED_RES[0]

    def get_real_world_coords(self, px, py, frame_w=None, frame_h=None):
        """
        Converts a single pixel-space centroid (px, py) -- in whatever resolution
        frame_w x frame_h it was extracted from (defaults to the native raw grid) --
        into real-world ground-plane coordinates in inches, relative to the origin
        directly beneath the camera lens.
        """
        if frame_w is None or frame_h is None:
            frame_w, frame_h = config.RAW_RES

        # Normalize to [0, 1], then to homogeneous image-plane coordinates
        u_norm, v_norm = px / frame_w, py / frame_h
        point_h = np.array([u_norm, v_norm, 1.0], dtype=np.float32)

        # Undistort/rectify a single point via the inverse camera matrix rather than
        # warping the whole frame
        undistorted_pts = cv2.undistortPoints(
            np.array([[[u_norm, v_norm]]], dtype=np.float32),
            self.K, self.dist, P=self.K
        )
        u_rect, v_rect = undistorted_pts[0, 0]

        # Map rectified normalized coordinates to physical inches via per-axis FOV trig
        real_x = config.CAMERA_HEIGHT_CM * np.tan((u_rect - 0.5) * self.fov_x_rad) * 0.393701
        real_y = config.CAMERA_HEIGHT_CM * np.tan((v_rect - 0.5) * self.fov_y_rad) * 0.393701
        return float(real_x), float(real_y)

    def rectify_full_frame(self, frame):
        """
        Retained only for the diagnostic dashboard, where a visually rectified preview
        image is useful to a human operator. Never called by the tracking/math pipeline --
        that path now rectifies single points via get_real_world_coords instead.
        """
        h, w = frame.shape[:2]
        K_px = self.K.copy()
        K_px[0, 0] *= w
        K_px[1, 1] *= h
        K_px[0, 2] *= w
        K_px[1, 2] *= h
        undistorted = cv2.undistort(frame, K_px, self.dist)
        return undistorted
