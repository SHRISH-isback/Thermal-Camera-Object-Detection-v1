import numpy as np

# --- PHYSICAL HARDWARE GEOMETRY ---
CAMERA_HEIGHT_CM = 75.0          
SENSOR_FOV_DEGREES = 110.0       
# Vertical FOV derived from the sensor's native 4:3 (32x24) aspect ratio
SENSOR_FOV_DEGREES_V = SENSOR_FOV_DEGREES * (24.0 / 32.0)

# --- PHYSICAL LAYOUT PLANE (INCHES) ---
REAL_WORLD_DIMS = (10.0, 10.0)    
UPSCALED_RES = (640, 480)          # Used ONLY for the diagnostic dashboard image, never the math pipeline
RAW_RES = (32, 24)                 # (width, height) -- native MLX90640 grid, the actual DSP/tracking domain

# --- RAW-DOMAIN <-> DISPLAY-DOMAIN SCALING ---
# The legacy pipeline ran every filtering stage on the upscaled 640x480 grid (307,200 px).
# The optimized pipeline performs all DSP/tracking math on the raw 32x24 grid (768 px) instead,
# a 400x reduction in elements processed per frame, and upscales only for the dashboard at the end.
RAW_TO_DISPLAY_SCALE = (UPSCALED_RES[0] / RAW_RES[0], UPSCALED_RES[1] / RAW_RES[1])
AREA_SCALE_FACTOR = (RAW_RES[0] * RAW_RES[1]) / (UPSCALED_RES[0] * UPSCALED_RES[1])  # 1/400

# --- DYNAMIC REFERENCE CALIBRATION ---
# Area configurations are referenced directly at the camera's actual mounting height
# (CAMERA_HEIGHT_CM) rather than extrapolated from a distant reference point. The previous
# REF_HEIGHT_CM=10.0 was tuned for a sensor mounted ~10cm above the target; with the real rig
# mounted at 75cm, the inverse-square height_scale = (REF_HEIGHT_CM/CAMERA_HEIGHT_CM)**2 in
# main.py/processing.py collapsed to ~0.018, driving current_min_spawn/current_min_keep down
# past the absolute floors below. With no effective area gate, single/double-pixel thermal
# noise and fragment contours were passing the area filter and spawning their own tracks --
# the cause of one physical target showing up as several overlapping boxes/IDs.
#
# Re-pointing REF_HEIGHT_CM at CAMERA_HEIGHT_CM makes height_scale == 1.0 at the real operating
# height (no extrapolation needed for the common case), while still preserving the inverse-
# square falloff if CAMERA_HEIGHT_CM is changed later for a different mount.
REF_HEIGHT_CM = CAMERA_HEIGHT_CM
REF_MIN_AREA_SPAWN = max(1.0, 2500 * AREA_SCALE_FACTOR)   # Large target footprint expected near lens
REF_MIN_AREA_KEEP = max(1.0, 800 * AREA_SCALE_FACTOR)

# Fallback limits to prevent settings from scaling to infinity or zero.
# Raised from the previous 0.5/0.65 (which sat below the real noise floor on a 768px raw grid
# and let near-empty contours through) to values that require a handful of contiguous warm
# pixels before a track can spawn -- run calibrate_area.py against your live setup to tune
# these precisely for your scene/lighting if needed.
ABSOLUTE_MIN_AREA_SPAWN = 45
ABSOLUTE_NOISE_FLOOR = 1.0

# --- PERSISTENCE FILTER GATING ---
# Minimum consecutive frames an object must be tracked before printing telemetry
MIN_TRACKING_AGE = 3             

# --- DEAD / OUTLIER PIXEL COMPENSATION (Step 2) ---
SENSOR_MIN_VALID_TEMP_C = -40.0
SENSOR_MAX_VALID_TEMP_C = 300.0
# Coordinates (row, col) of pixels flagged defective in the onboard EEPROM calibration table.
# Populated at runtime from the sensor driver when live hardware is connected.
KNOWN_DEAD_PIXELS = []

# --- TEMPORAL SMOOTHING: HOLT'S DOUBLE EXPONENTIAL SMOOTHING (Step 3) ---
# Replaces the legacy single EMA, which traded motion lag against noise suppression.
# DES tracks a trend term so it can suppress flicker without smearing moving targets.
DES_ALPHA = 0.15   # Data smoothing factor -- suppresses high-frequency thermal flicker
DES_GAMMA = 0.30   # Trend smoothing factor -- tracks genuine temperature change quickly
# Retained for reference / fallback simulation path
TEMPORAL_SMOOTHING_ALPHA = 0.35  

# --- DYNAMIC BACKGROUND MODEL: DUAL-RATE PIXEL-WISE UPDATE (Step 4) ---
# Replaces the legacy single-rate background model, which either absorbed stationary
# targets into the background (rate too fast) or failed to track drift (rate too slow).
#
# BUGFIX: BG_ETA_SLOW was previously 0.0001 -- small, but NOT zero. Any nonzero rate still
# asymptotically drags the background toward a perfectly stationary foreground target; it
# only changes how many minutes it takes, not whether it happens. That slow creep is what
# was causing tracked boxes to fade out and deregister after the camera (and target) sat
# still for a while. Foreground-classified pixels are now fully frozen (eta_slow == 0.0):
# the background for a pixel only ever updates again once that pixel returns to "drift"
# classification (delta < BG_DRIFT_THRESHOLD_C), e.g. because the target actually left.
BG_DRIFT_THRESHOLD_C = 1.5     # T_drift -- below this delta, a pixel is assumed to be ambient drift
BG_ETA_FAST = 0.05             # eta_fast -- applied to background-classified (drift) pixels
BG_ETA_SLOW = 0.0              # eta_slow -- applied to foreground-classified pixels: fully frozen
BACKGROUND_LEARNING_RATE = BG_ETA_FAST  # Retained name for any external references

# BUGFIX: BG_ETA_SLOW==0.0 only freezes a pixel for as long as it is *re-classified* as
# foreground on every single frame. The drift/foreground decision above was a single
# instantaneous comparison (delta < BG_DRIFT_THRESHOLD_C) re-evaluated from scratch each
# frame, with no memory of the previous classification. Ordinary sensor noise / DES settling
# means a perfectly stationary target's delta will occasionally flicker below
# BG_DRIFT_THRESHOLD_C for a single frame -- and the moment that happens, eta_fast nudges the
# background toward the target value, which *shrinks* the delta, making the next flicker even
# more likely. This is a self-reinforcing ratchet: a fixed target slowly gets absorbed into the
# background over the course of seconds-to-minutes (the "stops detecting after a while while
# fixated" symptom), even though each individual frame's threshold logic looked correct.
#
# Fixed with two changes:
#   1. A hysteresis gap between the threshold that *enters* the frozen/foreground state and the
#      (lower) threshold that *exits* it, so a delta that's still clearly target-like can't flip
#      the classification on a single noisy dip.
#   2. A consecutive-frame requirement before a frozen pixel is released back to drift-tracking,
#      so only a sustained, multi-frame return to ambient (i.e. the target actually left) can
#      unfreeze it -- not a single flickered low-delta frame.
BG_DRIFT_RELEASE_THRESHOLD_C = 0.6   # Must stay below THIS (lower) delta to be eligible for release
BG_RELEASE_CONSECUTIVE_FRAMES = 32   # ~2s at 16Hz of sustained low-delta before unfreezing

# Number of initial frames averaged (pixel-wise median) to seed the background model, instead
# of locking in whatever the very first frame happened to contain. Without this, booting the
# system with a target already in view bakes that target permanently into the background.
BACKGROUND_INIT_FRAMES = 10

# --- LOCAL SAUVOLA ADAPTIVE THRESHOLDING (Step 5) ---
# Replaces the legacy global median/std threshold, which collapses when a single hot
# environmental source skews the frame-wide statistics, masking colder/smaller targets.
#
# BUGFIX: SAUVOLA_R=128 assumed an 8-bit-normalized (0-255) signal, but this threshold runs
# directly on raw degrees-C delta values, which rarely exceed ~10-15C even for a strong
# target. With R that large, local_std/R was always ~0 regardless of scene content, which
# silently disabled Sauvola's local-contrast adaptivity and collapsed the formula to a flat
# threshold = local_mean * (1 - SAUVOLA_K) everywhere. At K=0.30 that flat threshold = 70% of
# local_mean -- loose enough that a wide warm halo around the real target (clothing, residual
# body heat in the air, etc.) cleared the bar too, inflating the contour well beyond the
# visually "hot" core and producing oversized bounding boxes. R is now sized to the actual
# delta range so the std term contributes meaningfully, and K is lowered so the flat-region
# fallback threshold sits closer to the local peak (90% of local_mean) instead of 70%.
SAUVOLA_WINDOW_SIZE = 5        # NxN sliding window, native raw-grid units
SAUVOLA_K = 0.10                # Sensitivity to local contrast -- lower = tighter/closer to peak
SAUVOLA_R = 6.0                 # Dynamic range of local std-dev, sized to real delta-C values (not 0-255)

# --- KALMAN FILTER STATE ESTIMATION (Step 9) ---
KALMAN_PROCESS_NOISE_INTENSITY = 0.5    # q -- continuous process noise intensity
KALMAN_MEASUREMENT_STD_X = 0.5          # sigma_x (inches) -- centroid measurement uncertainty
KALMAN_MEASUREMENT_STD_Y = 0.5          # sigma_y (inches)

# --- CONTOUR FRAGMENT MERGING (Step 6.5) ---
# A single physical heat source does not always threshold into one solid blob -- internal
# temperature gradients (a hotter face/hand next to a cooler torso, say) combined with the
# small Sauvola window can split it into 2+ disjoint contours in the same frame, each of
# which independently clears the area gate and spawns its own track ID. This was the root
# cause of "multiple boxes on one target." Any contours whose bounding boxes lie within
# CONTOUR_MERGE_GAP_PX raw-grid pixels of each other (touching/overlapping included) are
# merged into a single detection in processing.py before reaching the tracker. Keep this
# small relative to RAW_RES so two genuinely separate nearby objects don't get fused.
CONTOUR_MERGE_GAP_PX = 2

# --- TRACKING ASSOCIATION GATES (Step 10: Hungarian Algorithm) ---
# These are floors only -- main.py overrides them at startup with values derived from the
# sensor's actual real-world pixel pitch at the configured CAMERA_HEIGHT_CM/FOV (see the
# comment block in main() for why a fixed inch value here was too tight for a 32x24 sensor).
MAX_ASSOCIATION_DISTANCE = 5.5    
EDGE_MARGIN_PERCENT = 0.04
# Raised slightly from 16 -> 24 (~1.5s at 16Hz). With the background fade-out bug fixed,
# this mostly just adds margin against ordinary transient misses/occlusion; it is no longer
# papering over the fade-out, which is now fixed at its source above.
MAX_DISAPPEARED_FRAMES = 24        # N_vanish -- ~1.5 seconds at 16Hz before a lost track is pruned

# --- I2C BUS MANAGEMENT (Step 1) ---
I2C_FREQUENCY_HZ = 400000          # Elevated from the standard 100kHz to stabilize high-framerate transfers
MLX90640_I2C_ADDRESS = 0x33
I2C_HANG_REPEAT_LIMIT = 2          # Consecutive identical frames before a bus hang is declared

# --- MAXIMIZED FULL-VIEW CALIBRATION CORNERS (normalized 0-1, resolution independent) ---
CAMERA_SRC_POINTS_NORM = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

# --- RADIAL OPTICAL UNWARPING COORDINATES (normalized image-plane camera matrix) ---
# Principal point at frame center, focal length expressed as a fraction of frame width so the
# matrix stays valid whether it is applied to the raw 32x24 centroid or the 640x480 dashboard.
CAMERA_MATRIX_NORM = np.array([[0.78125, 0, 0.5],
                                [0, 0.78125, 0.5],
                                [0, 0, 1.0]], dtype=np.float32)
DIST_COEFFS = np.array([-0.1, 0.05, 0, 0, 0], dtype=np.float32)
DIAGNOSTIC_WRITE_INTERVAL_SEC = 1.0
