import cv2
import numpy as np
import config


class HeatDetector:
    """
    Signal Processing Core.

    Order-of-operations fix: the legacy pipeline upscaled the raw 24x32 grid to 640x480
    BEFORE filtering, forcing every downstream stage (background subtraction, thresholding,
    morphology) to run on a 307,200-element array. This version performs the entire DSP chain
    -- dead-pixel correction, temporal smoothing, background modeling, thresholding, and
    morphology -- directly on the native 32x24 (768-element) grid, a 400x reduction in work
    per frame, and only upscales once, at the very end, purely for the visual dashboard.
    """

    def __init__(self):
        self.learning_rate = config.BG_ETA_FAST
        self.background_model = None   # B_t : running dual-rate background model (raw grid)
        self.des_level = None          # S_t : DES smoothed level (raw grid)
        self.des_trend = None          # b_t : DES trend term (raw grid)
        self._bg_init_buffer = []      # Frames collected before the background model is seeded

        # Step 4b: per-pixel hysteresis state for the foreground freeze (see BUGFIX comment
        # in config.py next to BG_DRIFT_RELEASE_THRESHOLD_C). `frozen_mask` is the sticky
        # foreground/drift classification itself; `release_streak` counts consecutive frames
        # a currently-frozen pixel has spent back under the (lower) release threshold, so a
        # single noisy dip can't unfreeze it -- only a sustained return to ambient can.
        self.frozen_mask = None        # bool, raw grid: True == foreground/frozen
        self.release_streak = None     # int, raw grid: consecutive low-delta frames while frozen

    # ------------------------------------------------------------------
    # Step 2: Raw Dead-Pixel Compensation
    # ------------------------------------------------------------------
    def _correct_dead_pixels(self, raw_matrix):
        """Replaces EEPROM-flagged and runtime-detected outlier pixels with the median of
        their valid 3x3 neighborhood, before any spatial filtering can spread the defect."""
        matrix = raw_matrix.copy()
        h, w = matrix.shape

        invalid_mask = (
            np.isnan(matrix)
            | (matrix < config.SENSOR_MIN_VALID_TEMP_C)
            | (matrix > config.SENSOR_MAX_VALID_TEMP_C)
        )

        for (row, col) in config.KNOWN_DEAD_PIXELS:
            if 0 <= row < h and 0 <= col < w:
                invalid_mask[row, col] = True

        if not np.any(invalid_mask):
            return matrix

        bad_coords = np.argwhere(invalid_mask)
        for row, col in bad_coords:
            r0, r1 = max(0, row - 1), min(h, row + 2)
            c0, c1 = max(0, col - 1), min(w, col + 2)
            neighborhood = matrix[r0:r1, c0:c1]
            neighborhood_mask = invalid_mask[r0:r1, c0:c1]
            valid_values = neighborhood[~neighborhood_mask]
            if valid_values.size > 0:
                matrix[row, col] = np.median(valid_values)
            # If every neighbor is also invalid (rare), leave the prior corrected value in
            # place rather than injecting a fabricated number.

        return matrix

    # ------------------------------------------------------------------
    # Step 3: Temporal Noise Suppression via Double Exponential Smoothing
    # ------------------------------------------------------------------
    def _apply_double_exponential_smoothing(self, corrected_frame):
        """Holt's linear trend model. Decouples noise suppression (alpha) from
        motion-induced lag by tracking a per-pixel trend term (gamma), eliminating the
        cold 'ghosting' trail a single first-order EMA leaves behind moving targets."""
        if self.des_level is None:
            self.des_level = corrected_frame.copy()
            self.des_trend = np.zeros_like(corrected_frame)
            return self.des_level.copy()

        prev_level = self.des_level
        prev_trend = self.des_trend

        new_level = (config.DES_ALPHA * corrected_frame) + \
                    ((1 - config.DES_ALPHA) * (prev_level + prev_trend))
        new_trend = (config.DES_GAMMA * (new_level - prev_level)) + \
                    ((1 - config.DES_GAMMA) * prev_trend)

        self.des_level = new_level
        self.des_trend = new_trend
        return new_level.copy()

    # ------------------------------------------------------------------
    # Step 4: Dynamic Background Modeling via Dual-Rate Pixel-Wise Updates
    # ------------------------------------------------------------------
    def _update_background_model(self, smoothed_frame):
        """Each pixel updates at eta_fast when it looks like ambient drift and is fully frozen
        (eta_slow == 0.0) when it looks like an occupied foreground pixel, so a stationary warm
        target is never absorbed into the background ('fade-out'). The background only resumes
        updating for a pixel once it genuinely returns to drift.

        BUGFIX (background-absorption ratchet): the foreground/drift decision used to be a
        single instantaneous `delta < BG_DRIFT_THRESHOLD_C` comparison, re-evaluated from
        scratch every frame. A perfectly stationary target's delta naturally flickers a little
        frame to frame (sensor noise, DES settling) -- one frame dipping under the threshold
        was enough to flip that pixel back to "drift", nudge the background toward the target
        value at eta_fast, shrink the delta further, and make the next flicker more likely.
        That self-reinforcing ratchet is what slowly absorbed a perfectly stationary target
        into the background over time, even with eta_slow pinned to 0.0.
        Fixed by making the freeze STICKY: entering frozen state still uses
        BG_DRIFT_THRESHOLD_C, but leaving it requires the delta to stay below a lower
        BG_DRIFT_RELEASE_THRESHOLD_C for BG_RELEASE_CONSECUTIVE_FRAMES in a row (classic
        Schmitt-trigger hysteresis + debounce). A single noisy dip can no longer unfreeze a
        pixel; only a sustained, multi-frame return to ambient (the target actually leaving)
        can.

        BUGFIX: the model used to seed itself from a single first frame, which (a) bakes in
        whatever noise that one frame happened to contain, and (b) permanently absorbs a
        target if one is already in view at boot. It now collects config.BACKGROUND_INIT_FRAMES
        frames and seeds from their pixel-wise median instead.
        """
        if self.background_model is None:
            self._bg_init_buffer.append(smoothed_frame.copy())
            if len(self._bg_init_buffer) < config.BACKGROUND_INIT_FRAMES:
                # Not enough samples yet to trust a seed -- report zero delta for this frame
                # (no detections) rather than locking in a single noisy/possibly-occupied frame.
                return smoothed_frame.copy()
            self.background_model = np.median(
                np.stack(self._bg_init_buffer, axis=0), axis=0
            ).astype(np.float32)
            self._bg_init_buffer = []
            self.frozen_mask = np.zeros_like(self.background_model, dtype=bool)
            self.release_streak = np.zeros_like(self.background_model, dtype=np.int32)
            return self.background_model.copy()

        delta = np.abs(smoothed_frame - self.background_model)

        # Newly-foreground pixels (not yet frozen) latch into the frozen state the moment they
        # clear the entry threshold -- same trigger condition as before.
        newly_foreground = (~self.frozen_mask) & (delta >= config.BG_DRIFT_THRESHOLD_C)
        self.frozen_mask = self.frozen_mask | newly_foreground

        # Among currently-frozen pixels, track how many CONSECUTIVE frames each has spent back
        # under the (lower) release threshold. Any frame that breaks the streak resets it to 0,
        # so only a sustained run counts -- a single flicker back above the release threshold
        # cancels the progress made so far.
        below_release = delta < config.BG_DRIFT_RELEASE_THRESHOLD_C
        self.release_streak = np.where(
            self.frozen_mask & below_release, self.release_streak + 1, 0
        )

        # Unfreeze only once the streak requirement is met; reset its counter on release.
        ready_to_release = self.frozen_mask & (self.release_streak >= config.BG_RELEASE_CONSECUTIVE_FRAMES)
        self.frozen_mask = self.frozen_mask & (~ready_to_release)
        self.release_streak = np.where(ready_to_release, 0, self.release_streak)

        eta = np.where(self.frozen_mask, config.BG_ETA_SLOW, config.BG_ETA_FAST).astype(np.float32)
        self.background_model = ((1.0 - eta) * self.background_model) + (eta * smoothed_frame)
        return self.background_model.copy()

    # ------------------------------------------------------------------
    # Step 5: Local Sauvola Adaptive Thresholding
    # ------------------------------------------------------------------
    def _sauvola_threshold(self, delta_frame):
        """Computes a per-pixel binarization threshold from the local mean/std-dev in a
        sliding NxN window, instead of one global statistic. This prevents a single hot
        environmental source from skewing the frame-wide threshold and masking smaller or
        cooler targets elsewhere in the frame (the 'threshold collapse' failure mode)."""
        win = config.SAUVOLA_WINDOW_SIZE
        delta_frame = delta_frame.astype(np.float32)

        local_mean = cv2.boxFilter(delta_frame, ddepth=-1, ksize=(win, win), normalize=True)
        local_sq_mean = cv2.boxFilter(delta_frame * delta_frame, ddepth=-1, ksize=(win, win), normalize=True)
        local_var = np.clip(local_sq_mean - (local_mean * local_mean), 0, None)
        local_std = np.sqrt(local_var)

        threshold_map = local_mean * (1.0 + config.SAUVOLA_K * ((local_std / config.SAUVOLA_R) - 1.0))

        # Height-based floor: as installation height grows, thermal signal dilutes per-pixel,
        # so the absolute noise floor is allowed to relax proportionally.
        height_ratio = config.REF_HEIGHT_CM / config.CAMERA_HEIGHT_CM
        dynamic_noise_floor = max(config.ABSOLUTE_NOISE_FLOOR, 1.5 * height_ratio)
        threshold_map = np.maximum(threshold_map, dynamic_noise_floor)

        binary_mask = np.where(delta_frame > threshold_map, 255, 0).astype(np.uint8)
        return binary_mask

    # ------------------------------------------------------------------
    # Step 6: Morphological Cleansing (native 32x24 domain)
    # ------------------------------------------------------------------
    def _morphological_cleanse(self, binary_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    # ------------------------------------------------------------------
    # Step 6.5: Fragment Merging -- collapse split contours back into one detection
    # ------------------------------------------------------------------
    @staticmethod
    def merge_fragmented_contours(contours, mask_shape, gap_px):
        """
        Groups contours that almost certainly belong to the same physical heat source.

        One warm object does not always threshold into a single solid blob: internal
        temperature gradients (a hotter face/hand vs a cooler torso) interacting with the
        small Sauvola window can split it into several disjoint contours in the same frame.
        Each fragment independently clears the area gate downstream and spawns its own
        track ID -- this is the mechanism behind "multiple boxes on one target."

        Any contours whose bounding boxes are within `gap_px` raw-grid pixels of one another
        (touching/overlapping included) are merged via union-find into a single combined mask
        and re-extracted as one contour, so the rest of the pipeline only ever sees one
        detection per physical blob. Two genuinely separate objects further apart than
        `gap_px` are left as independent detections.

        Returns a list of (merged_contour, merged_mask) tuples.
        """
        n = len(contours)
        if n == 0:
            return []
        if n == 1:
            solo_mask = np.zeros(mask_shape, dtype=np.uint8)
            cv2.drawContours(solo_mask, [contours[0]], -1, 255, -1)
            return [(contours[0], solo_mask)]

        boxes = [cv2.boundingRect(c) for c in contours]

        def boxes_within_gap(a, b, gap):
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            ax0, ay0, ax1, ay1 = ax - gap, ay - gap, ax + aw + gap, ay + ah + gap
            return not (ax1 < bx or bx + bw < ax0 or ay1 < by or by + bh < ay0)

        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                if boxes_within_gap(boxes[i], boxes[j], gap_px):
                    union(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        merged = []
        for indices in groups.values():
            if len(indices) == 1:
                idx = indices[0]
                solo_mask = np.zeros(mask_shape, dtype=np.uint8)
                cv2.drawContours(solo_mask, [contours[idx]], -1, 255, -1)
                merged.append((contours[idx], solo_mask))
                continue

            group_mask = np.zeros(mask_shape, dtype=np.uint8)
            for idx in indices:
                cv2.drawContours(group_mask, [contours[idx]], -1, 255, -1)

            regrouped, _ = cv2.findContours(group_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not regrouped:
                continue
            # By construction this should be one connected region; in the rare case the
            # union-find chain bridged through a third intermediary into >1 disjoint piece,
            # keep only the largest so a stray sliver can't masquerade as the merged target.
            biggest = max(regrouped, key=cv2.contourArea)
            merged.append((biggest, group_mask))

        return merged

    # ------------------------------------------------------------------
    def process_frame(self, raw_matrix):
        """
        Runs the full raw-domain DSP chain and returns:
          fine_thermal  -- corrected/smoothed temperature grid, native 32x24 resolution
          binary_mask   -- cleaned foreground mask, native 32x24 resolution
        Both stay at sensor resolution; main.py upscales only for the dashboard image.
        """
        raw_float = raw_matrix.astype(np.float32)

        corrected = self._correct_dead_pixels(raw_float)
        smoothed = self._apply_double_exponential_smoothing(corrected)
        background = self._update_background_model(smoothed)

        raw_diff = cv2.absdiff(smoothed, background)
        binary_mask = self._sauvola_threshold(raw_diff)
        binary_mask = self._morphological_cleanse(binary_mask)

        return smoothed, binary_mask
