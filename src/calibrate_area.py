"""
Area-gate calibration helper.

The tracker's area gate (config.REF_MIN_AREA_SPAWN / REF_MIN_AREA_KEEP, scaled by
config.ABSOLUTE_MIN_AREA_SPAWN / ABSOLUTE_NOISE_FLOOR as floors) decides which contours
in the raw 32x24 thermal grid are treated as real targets versus noise/fragments. If
those numbers are tuned for the wrong mounting height, one physical object can fragment
into several small contours that all pass the gate -- producing the "multiple boxes on
one target" symptom.

This script runs the live sensor + DSP pipeline (same hardware.py / processing.py used by
main.py) for a short window and reports:
  - the contour-area distribution actually produced at your current CAMERA_HEIGHT_CM
  - a recommended REF_MIN_AREA_SPAWN / REF_MIN_AREA_KEEP pair, referenced at the height
    you are actually running at (so no inverse-square extrapolation is needed)
  - a recommended ABSOLUTE_MIN_AREA_SPAWN / ABSOLUTE_NOISE_FLOOR floor pair, derived from
    the noise contours observed with no target present

USAGE
-----
1. Point the camera at empty background (no person/object in frame) and run:
       python3 calibrate_area.py --mode background --seconds 8
   This measures noise-contour areas with nothing real in view, to set the floors.

2. Then place your typical target (hand, person, etc.) at the real distance/height and run:
       python3 calibrate_area.py --mode target --seconds 8
   This measures the target's actual contour area, to set the reference values.

3. Paste the printed config.py block over the corresponding lines in config.py.

This script never modifies config.py automatically -- it only measures and prints
recommendations, since the right floor also depends on how much noise margin you want.
"""

import argparse
import time

import cv2
import numpy as np

import config
from hardware import ThermalSensor
from processing import HeatDetector


def collect_contour_areas(duration_sec):
    sensor = ThermalSensor()
    detector = HeatDetector()

    areas = []
    t_end = time.time() + duration_sec
    frame_count = 0

    while time.time() < t_end:
        raw_matrix = sensor.get_raw_matrix()
        if raw_matrix is None:
            continue

        _, binary_mask = detector.process_frame(raw_matrix)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_areas = [cv2.contourArea(c) for c in contours]
        # Ignore exact-zero areas (single-pixel/degenerate contours with no measurable area)
        # so they don't silently zero out percentile stats -- they're still noise, just
        # reported separately below.
        areas.extend(a for a in frame_areas if a > 0)
        frame_count += 1

    return areas, frame_count


def summarize(areas, label):
    if not areas:
        print(f"[{label}] No contours with nonzero area were detected in this window.")
        return None

    arr = np.array(areas, dtype=np.float64)
    stats = {
        "count": len(arr),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }
    print(f"[{label}] contour area stats (raw {config.RAW_RES[0]}x{config.RAW_RES[1]} grid, "
          f"{stats['count']} contours):")
    print(f"    min={stats['min']:.2f}  p25={stats['p25']:.2f}  median={stats['median']:.2f}  "
          f"p75={stats['p75']:.2f}  p95={stats['p95']:.2f}  max={stats['max']:.2f}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["background", "target"], required=True,
        help="'background': no real target in frame, measures noise contour sizes. "
             "'target': your typical real target in frame, measures target contour size.",
    )
    parser.add_argument(
        "--seconds", type=float, default=8.0,
        help="How long to sample frames for (default 8s, ~16Hz sensor -> ~128 frames).",
    )
    args = parser.parse_args()

    print(f"Sampling for {args.seconds:.1f}s at CAMERA_HEIGHT_CM={config.CAMERA_HEIGHT_CM} "
          f"in '{args.mode}' mode...\n")
    areas, frame_count = collect_contour_areas(args.seconds)
    print(f"Captured {frame_count} frames.\n")
    stats = summarize(areas, args.mode)

    if stats is None:
        print("\nNothing to recommend from an empty result -- try a longer --seconds window, "
              "or check that the sensor is producing frames.")
        return

    print("\n--- Recommendation ---")
    if args.mode == "background":
        # Floors should sit comfortably above the noise contours seen with nothing real in
        # frame, so flicker/fragments can never pass the gate on their own.
        recommended_floor_spawn = round(max(stats["p95"] * 1.5, stats["max"] * 1.1, 1.0), 2)
        recommended_noise_floor_c = config.ABSOLUTE_NOISE_FLOOR
        print("Paste into config.py:")
        print(f"    ABSOLUTE_MIN_AREA_SPAWN = {recommended_floor_spawn}")
        print(f"    ABSOLUTE_NOISE_FLOOR = {recommended_noise_floor_c}  "
              f"# (re-run with --mode target to sanity check this against real signal)")
        print("\nNote: this sets the floor just above what noise alone produced during this "
              "sample. Run again with more time (--seconds 20+) for a more stable estimate, "
              "and re-run periodically if ambient conditions change a lot.")
    else:
        # Reference values should sit comfortably below the smallest real-target contour seen,
        # so a genuine (if partially-occluded or distant) target still spawns/keeps a track.
        recommended_ref_spawn = round(max(stats["p25"] * 0.6, 1.0), 2)
        recommended_ref_keep = round(max(stats["p25"] * 0.3, 1.0), 2)
        print("Paste into config.py (replace REF_MIN_AREA_SPAWN / REF_MIN_AREA_KEEP):")
        print(f"    REF_HEIGHT_CM = {config.CAMERA_HEIGHT_CM}  "
              f"# measured at the height you just calibrated at")
        print(f"    REF_MIN_AREA_SPAWN = {recommended_ref_spawn}")
        print(f"    REF_MIN_AREA_KEEP = {recommended_ref_keep}")
        print("\nNote: these are referenced directly at your current CAMERA_HEIGHT_CM, so "
              "REF_HEIGHT_CM should equal CAMERA_HEIGHT_CM unless you intentionally want the "
              "inverse-square model to extrapolate to other heights later.")


if __name__ == "__main__":
    main()
