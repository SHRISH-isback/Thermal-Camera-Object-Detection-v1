import time
import cv2
import numpy as np
import config
from hardware import ThermalSensor
from processing import HeatDetector
from spatial import SpatialTransformer
from tracker import CentroidTracker


def main():
    sensor = ThermalSensor()
    detector = HeatDetector()
    transformer = SpatialTransformer()
    tracker = CentroidTracker()

    raw_w, raw_h = config.RAW_RES
    disp_w, disp_h = config.UPSCALED_RES
    margin_w = max(1, int(raw_w * config.EDGE_MARGIN_PERCENT))
    margin_h = max(1, int(raw_h * config.EDGE_MARGIN_PERCENT))

    # BUGFIX: MAX_ASSOCIATION_DISTANCE and KALMAN_MEASUREMENT_STD_X/Y were fixed inch values
    # tuned for a much higher-resolution centroid estimate. At this rig's actual mounting
    # height, one raw 32x24 sensor pixel covers several inches of real floor space (a function
    # of CAMERA_HEIGHT_CM and SENSOR_FOV_DEGREES) -- so a perfectly normal 1-2 raw-pixel
    # frame-to-frame jitter in the centroid (just from which marginal pixels cross threshold
    # that frame) was *larger* than the association gate. The Hungarian matcher would reject
    # the match, the existing track would freeze at its last position via Kalman prediction
    # (a second, stale box), and a brand new track would spawn for the "unmatched" detection --
    # this is what was causing duplicate boxes AND boxes appearing to vanish (the frozen old
    # track ages out after MAX_DISAPPEARED_FRAMES while a new ID keeps getting spawned in its
    # place). These gates are now derived from the sensor's actual real-world pixel pitch at
    # the configured height/FOV, so they scale correctly for any mounting height instead of
    # silently being too tight (or too loose) for whatever resolution the rig happens to have.
    raw_px_width_in = transformer.view_width_inches / raw_w
    raw_px_height_in = transformer.view_height_inches / raw_h
    avg_raw_px_in = (raw_px_width_in + raw_px_height_in) / 2.0

    # Allow ~3 raw pixels of predicted-vs-detected slack before refusing a match -- enough to
    # absorb normal quantization jitter without merging two genuinely separate nearby targets.
    config.MAX_ASSOCIATION_DISTANCE = max(config.MAX_ASSOCIATION_DISTANCE, 3.0 * avg_raw_px_in)
    # Measurement noise should reflect that a centroid can't be pinpointed more precisely than
    # roughly half a raw pixel's worth of real-world distance.
    config.KALMAN_MEASUREMENT_STD_X = max(config.KALMAN_MEASUREMENT_STD_X, 0.6 * raw_px_width_in)
    config.KALMAN_MEASUREMENT_STD_Y = max(config.KALMAN_MEASUREMENT_STD_Y, 0.6 * raw_px_height_in)

    print(f"[INFO] Derived tracking gates for {config.CAMERA_HEIGHT_CM}cm mount: "
          f"~{avg_raw_px_in:.2f}in/raw-px -> MAX_ASSOCIATION_DISTANCE={config.MAX_ASSOCIATION_DISTANCE:.2f}in, "
          f"KALMAN_STD=({config.KALMAN_MEASUREMENT_STD_X:.2f}, {config.KALMAN_MEASUREMENT_STD_Y:.2f})in")

    print("\n" + "=" * 65)
    print("  Unified Object Coordinate Tracker Core Booted")
    print("  Raw-Domain DSP | Kalman-Filtered Hungarian Tracking Connected")
    print("=" * 65 + "\n")

    time.sleep(1.0)
    prev_frame_time = time.time()
    last_diagnostic_write = 0.0

    try:
        while True:
            raw_matrix = sensor.get_raw_matrix()
            if raw_matrix is None:
                continue

            transformer.recalibrate_geometry()

            # Entire DSP chain (dead-pixel correction -> DES -> dual-rate background ->
            # Sauvola threshold -> morphology) runs on the native 32x24 grid -- 768
            # elements per frame instead of the legacy 307,200.
            fine_thermal, binary_mask = detector.process_frame(raw_matrix)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Step 6.5: collapse contours that fragmented from the same physical heat source
            # (internal temperature gradients + the small Sauvola window can split one warm
            # object into several disjoint contours) back into single detections, BEFORE area
            # gating and tracking ever see them. This is what stops one target from spawning
            # multiple overlapping boxes/IDs.
            merged_groups = detector.merge_fragmented_contours(
                contours, binary_mask.shape, config.CONTOUR_MERGE_GAP_PX
            )

            raw_detections = []

            # Dynamic target-size expectations based on the physical inverse-square law,
            # rescaled into the native 32x24 raw-grid area units (see config.AREA_SCALE_FACTOR).
            height_scale = (config.REF_HEIGHT_CM / config.CAMERA_HEIGHT_CM) ** 2
            current_min_spawn = max(config.ABSOLUTE_MIN_AREA_SPAWN, config.REF_MIN_AREA_SPAWN * height_scale)
            current_min_keep = max(config.ABSOLUTE_MIN_AREA_SPAWN / 2.0, config.REF_MIN_AREA_KEEP * height_scale)

            for cnt, group_mask in merged_groups:
                area = cv2.contourArea(cnt)
                rx, ry, rw, rh = cv2.boundingRect(cnt)

                if rx < margin_w or ry < margin_h or (rx + rw) > (raw_w - margin_w) or (ry + rh) > (raw_h - margin_h):
                    if area < (current_min_spawn * 2.0):
                        continue

                # Step 7: contour moments computed directly on the raw 32x24 binary mask
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx_px = M["m10"] / M["m00"]
                cy_px = M["m01"] / M["m00"]

                # Localized spatial hysteresis against existing active tracks (raw-grid units)
                near_existing_track = False
                pad = max(1, int(40 * (raw_w / disp_w)))
                for obj_id, data in tracker.objects.items():
                    bx, by, bw, bh = data["bbox"]
                    if (bx - pad <= cx_px <= bx + bw + pad) and (by - pad <= cy_px <= by + bh + pad):
                        near_existing_track = True
                        break

                if near_existing_track:
                    if area < current_min_keep:
                        continue
                else:
                    if area < current_min_spawn:
                        continue

                # Step 8: point-based projection -- rectify/translate this single centroid
                # rather than warping the whole frame.
                cx_in, cy_in = transformer.get_real_world_coords(cx_px, cy_px, raw_w, raw_h)

                # Use the merged group's mask (not just the single sub-contour) so the
                # representative-temperature sample covers the full physical target.
                contour_temps = fine_thermal[group_mask > 0]
                contour_bg = detector.background_model[group_mask > 0]

                if len(contour_temps) > 0:
                    abs_deltas = np.abs(contour_temps - contour_bg)
                    peak_delta_idx = np.argmax(abs_deltas)
                    representative_temp = contour_temps[peak_delta_idx]
                else:
                    representative_temp = 0.0

                raw_detections.append([
                    float(cx_in), float(cy_in),
                    int(rx), int(ry), int(rw), int(rh),
                    float(representative_temp)
                ])

            # Step 9 + 10: Kalman-predicted states, globally matched via the Hungarian algorithm.
            tracked_objects = tracker.update(raw_detections)
            current_time = time.time()

            confirmed_objects = {}
            frame_output_lines = []

            for obj_id, data in tracked_objects.items():
                if data["hits"] >= config.MIN_TRACKING_AGE:
                    cx_in, cy_in = data["centroid"]
                    confirmed_objects[obj_id] = data

                    x_val, y_val = round(float(cx_in), 2), round(float(cy_in), 2)
                    t_val = round(float(data["temperature"]), 1)
                    frame_output_lines.append(
                        f"[ID {obj_id:02d}] X: {x_val:5.2f} in | Y: {y_val:5.2f} in | Temp: {t_val:4.1f}°C"
                    )

            if len(frame_output_lines) > 0:
                print(f"\n--- Timestamp: {current_time:.2f} ---")
                for line in frame_output_lines:
                    print(line)

            # --- UI Render Engine ---
            # The raw 32x24 thermal/mask grids are upscaled ONLY here, for the human-facing
            # dashboard image. This single bicubic step is isolated from the tracking loop
            # and never feeds back into detection or coordinate math.
            f_min, f_max = np.min(fine_thermal), np.max(fine_thermal)
            if (f_max - f_min) < 2.0:
                f_max = f_min + 2.0
            norm_thermal = np.clip(fine_thermal, f_min, f_max)
            norm_thermal = cv2.normalize(norm_thermal, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            norm_thermal = cv2.resize(norm_thermal, config.UPSCALED_RES, interpolation=cv2.INTER_CUBIC)

            display_frame = cv2.applyColorMap(norm_thermal, cv2.COLORMAP_JET)
            # BUGFIX: this used to call transformer.rectify_full_frame(display_frame) here,
            # warping the displayed image via cv2.undistort -- but the box overlay below is
            # plotted with a plain linear (raw_px * RAW_TO_DISPLAY_SCALE) mapping that does NOT
            # go through that same warp. The background image and the box were therefore in two
            # different geometric spaces: every box looked shifted/mis-sized relative to the
            # actual heat signature, worse the further off-center the target was. The math
            # pipeline never used rectify_full_frame anyway (see spatial.py) -- it only existed
            # for this dashboard preview -- so the simplest correct fix is to stop distorting the
            # displayed image and keep the dashboard in the same raw linear-pixel space as the
            # box coordinates. (If true lens-undistorted preview is wanted later, the box
            # corners must be pushed through the identical cv2.undistortPoints mapping before
            # drawing, not drawn on top of an already-warped frame.)
            workspace_canvas = np.full((disp_h, disp_w, 3), 245, dtype=np.uint8)

            for i in range(1, 10):
                grid_x, grid_y = int((i / 10.0) * disp_w), int((i / 10.0) * disp_h)
                cv2.line(workspace_canvas, (grid_x, 0), (grid_x, disp_h), (225, 225, 225), 1)
                cv2.line(workspace_canvas, (0, grid_y), (disp_w, grid_y), (225, 225, 225), 1)

            scale_x, scale_y = config.RAW_TO_DISPLAY_SCALE

            for obj_id, data in confirmed_objects.items():
                cx_in, cy_in = data["centroid"]
                rx, ry, rw, rh = data["bbox"]

                drx, dry = int(rx * scale_x), int(ry * scale_y)
                drw, drh = int(rw * scale_x), int(rh * scale_y)
                cv2.rectangle(display_frame, (drx, dry), (drx + drw, dry + drh), (255, 255, 255), 1)

                half_w = transformer.view_width_inches / 2
                half_h = transformer.view_height_inches / 2
                px_cx = int(((cx_in + half_w) / transformer.view_width_inches) * disp_w)
                px_cy = int(((cy_in + half_h) / transformer.view_height_inches) * disp_h)

                if 0 <= px_cx < disp_w and 0 <= px_cy < disp_h:
                    cv2.circle(workspace_canvas, (px_cx, px_cy), 6, (0, 140, 255), -1)
                    cv2.putText(workspace_canvas, f"ID {obj_id:02d}: {data['temperature']:.1f}C",
                                (px_cx + 10, px_cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1, cv2.LINE_AA)

            fps = 1.0 / (current_time - prev_frame_time) if current_time > prev_frame_time else 0.0
            prev_frame_time = current_time
            cv2.putText(display_frame, f"Thermal Field | {fps:.1f} FPS", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(workspace_canvas, f"Spatial Canvas ({config.CAMERA_HEIGHT_CM}cm)", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (75, 75, 75), 1, cv2.LINE_AA)

            diagnostic_panel = np.hstack([display_frame, workspace_canvas])
            if current_time - last_diagnostic_write >= config.DIAGNOSTIC_WRITE_INTERVAL_SEC:
                cv2.imwrite('live_view.jpg', diagnostic_panel)
                last_diagnostic_write = current_time

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[HALT] Tracking program terminated.")


if __name__ == "__main__":
    main()
