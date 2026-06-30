import numpy as np
import time
from scipy.optimize import linear_sum_assignment
import config


class KalmanTrack2D:
    """
    Step 9: independent 2D constant-velocity Kalman filter per tracked target.

    State vector x = [px, py, vx, vy] (inches, inches/sec). Predicting forward every
    frame -- even across a missed detection -- keeps trajectories smooth and lets the
    tracker bridge brief dropouts using the estimated velocity instead of freezing or
    snapping the reported position.
    """

    def __init__(self, px, py, timestamp):
        self.x = np.array([px, py, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4) * 10.0
        self.last_update_time = timestamp

        sx = config.KALMAN_MEASUREMENT_STD_X
        sy = config.KALMAN_MEASUREMENT_STD_Y
        self.R = np.array([[sx ** 2, 0.0], [0.0, sy ** 2]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.q = config.KALMAN_PROCESS_NOISE_INTENSITY

    def _build_F_Q(self, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        q = self.q
        Q = q * np.array([
            [dt ** 3 / 3, 0, dt ** 2 / 2, 0],
            [0, dt ** 3 / 3, 0, dt ** 2 / 2],
            [dt ** 2 / 2, 0, dt, 0],
            [0, dt ** 2 / 2, 0, dt],
        ], dtype=np.float64)
        return F, Q

    def predict(self, timestamp):
        dt = max(1e-3, timestamp - self.last_update_time)
        F, Q = self._build_F_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_update_time = timestamp
        return self.x[0], self.x[1]

    def update(self, zx, zy):
        z = np.array([zx, zy], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self):
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self):
        return float(np.hypot(self.x[2], self.x[3]))


class CentroidTracker:
    """
    Spatiotemporal Track Manager.

    Step 10: target association is now solved globally via the Hungarian algorithm
    instead of a greedy nearest-distance sort, which prevents the chain-reaction
    mis-assignments that fragment IDs when two targets are near each other and a
    sub-optimal early greedy match locks out a better later one.
    """

    def __init__(self, max_disappeared=None):
        self.next_object_id = 1
        self.objects = {}
        self.disappeared = {}
        self.kalman_filters = {}
        self.max_disappeared = max_disappeared or config.MAX_DISAPPEARED_FRAMES

    def register(self, centroid, bbox, temp, timestamp):
        oid = self.next_object_id
        self.kalman_filters[oid] = KalmanTrack2D(centroid[0], centroid[1], timestamp)
        self.objects[oid] = {
            "centroid": centroid,
            "bbox": bbox,
            "temperature": temp,
            "speed": 0.0,
            "last_seen": timestamp,
            "hits": 1,
        }
        self.disappeared[oid] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.kalman_filters[object_id]

    def update(self, rects):
        timestamp = time.time()

        # Predict every active track forward first, regardless of whether it gets matched
        # this frame -- this is what lets the tracker bridge transient dropouts smoothly.
        predicted_positions = {
            oid: kf.predict(timestamp) for oid, kf in self.kalman_filters.items()
        }

        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                else:
                    px, py = predicted_positions[object_id]
                    self.objects[object_id]["centroid"] = (px, py)
            return self.objects

        input_centroids = np.array([r[:2] for r in rects], dtype=np.float64)

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(tuple(rects[i][:2]), rects[i][2:6], rects[i][6], timestamp)
            return self.objects

        object_ids = list(self.objects.keys())
        predicted_centroids = np.array([predicted_positions[oid] for oid in object_ids])

        # Global bipartite cost matrix: Euclidean distance between each predicted track
        # position and each newly extracted contour centroid.
        diffs = predicted_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :]
        cost_matrix = np.linalg.norm(diffs, axis=2)

        # Hungarian algorithm solves the assignment in O(n^3), minimizing total matched distance.
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        used_rows, used_cols = set(), set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] > config.MAX_ASSOCIATION_DISTANCE:
                continue  # Distance-gated: reject matches that are implausibly far apart

            object_id = object_ids[row]
            prev_data = self.objects[object_id]

            kf = self.kalman_filters[object_id]
            zx, zy = rects[col][:2]
            kf.update(zx, zy)
            new_centroid = kf.position

            self.objects[object_id] = {
                "centroid": new_centroid,
                "bbox": rects[col][2:6],
                "temperature": rects[col][6],
                "speed": kf.speed,
                "last_seen": timestamp,
                "hits": prev_data["hits"] + 1,
            }
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing tracks: keep their Kalman-predicted state, increment vanish counter
        for row in set(range(len(object_ids))).difference(used_rows):
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
            else:
                px, py = predicted_positions[object_id]
                self.objects[object_id]["centroid"] = (px, py)

        # Unmatched detections: spawn new tentative tracks
        for col in set(range(len(rects))).difference(used_cols):
            self.register(tuple(rects[col][:2]), rects[col][2:6], rects[col][6], timestamp)

        return self.objects
