# Thermal Camera Object Detection

A deterministic, privacy-preserving, real-time tracking pipeline for a low-resolution infrared thermal array on edge hardware. It detects warm objects (for example people) in a 32×24 thermal image, tracks each one with a persistent ID, and reports where it is on the floor (in inches) and how warm it is, all without a neural network and without capturing anything that could identify a person.

Runs on a Raspberry Pi 5 with a Melexis **MLX90640** sensor at about **7.8 FPS** (sensor limit: 16 Hz).


## Why this exists

RGB cameras with CNN or transformer detectors are accurate but capture identifiable faces and need GPU-class compute. A 768-pixel thermal grid cannot record identity by construction, and classical signal processing is cheap enough for a low-power board. The goal is anonymous occupancy and spatial telemetry (smart buildings, ambient care, physical security) with predictable behaviour you can reason about, because every step is a fixed algorithm rather than a learned model.

## What it does

```
MLX90640 (32x24, 16 Hz, I2C)
  -> dead / outlier pixel fix
  -> Holt double-exponential smoothing (per pixel)
  -> median-seeded dual-rate background model with hysteresis
  -> local Sauvola threshold on |frame - background|
  -> morphological open + close
  -> fragmented-contour merging
  -> area gate (spawn / keep, scaled by mounting height)
  -> Kalman filter + Hungarian assignment (persistent IDs)
  -> lens undistortion + trigonometric ground projection (inches)
  -> console telemetry + diagnostic dashboard image
```

Console output for each confirmed track:

```
[ID 01] X: 12.4 in | Y: -3.1 in | Temp: 31.2°C
```

A diagnostic image (`live_view.jpg`) is written about once per second: the thermal view with one bounding marker per target, next to a floor-plane canvas showing each target's position, ID and temperature.

If the sensor library or hardware is not present, the program runs a **synthetic simulation** (two warm targets and one cool target on sinusoidal paths), so you can try the whole pipeline without a sensor.

## Design choices

- **Everything runs on the native 32×24 grid.** The 640×480 image exists only for the human-readable dashboard and never feeds detection, tracking or geometry. That is about 400× fewer elements per frame than a display-resolution pipeline.
- **Only blob centroids are corrected for lens distortion**, not whole frames. Each centroid is undistorted with `cv2.undistortPoints`, then projected onto the floor with per-axis tangent geometry from the camera height and field of view. There is no full-frame warp and no homography.
- **Background model.** A single fixed learning rate either absorbs people who stand still or fails to follow ambient drift. Each pixel is classified as drifting background or foreground and updated at its own rate; foreground pixels are frozen (rate exactly 0). Leaving the frozen state needs a sustained return to ambient (hysteresis: enter at 1.5 °C, release at 0.6 °C for 32 frames), so sensor noise cannot slowly erode a stationary target.
- **Area gates scale with mounting height** (inverse-square, from a reference height), so the same code works at different heights after running the calibration helper.
- **Tracking.** Constant-velocity Kalman filter per target with global Hungarian assignment instead of greedy matching, so crossing targets keep their identities. The association distance and measurement noise are derived from the real-world size of one sensor pixel at the configured height.

## Hardware

| Part | Detail |
|---|---|
| Sensor | Melexis MLX90640, 32×24 pixels, 110° field of view (about 82.5° vertical), I²C address `0x33` |
| Host | Raspberry Pi 5 (developed and measured on this board; other Pis with I²C should work but are untested) |
| Mounting | Looking down at the floor, default height 75 cm (`CAMERA_HEIGHT_CM` in `src/config.py`) |
| Bus | I²C, 400 kHz requested; sensor refresh 16 Hz |

Wire the sensor to the Pi's 3.3 V, GND, SDA and SCL pins and enable I²C (`sudo raspi-config` → Interface Options → I2C).

## Setup

```bash
git clone https://github.com/SHRISH-isback/Thermal-Camera-Object-Detection-v1.git
cd Thermal-Camera-Object-Detection-v1
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy opencv-python adafruit-blinka adafruit-circuitpython-mlx90640
```

On a machine without the sensor, only `numpy`, `scipy` and `opencv-python` are needed; the program falls back to the simulation.

## Run

```bash
python src/main.py
```

The program takes the first 10 frames to build the background model, so keep the scene empty at start-up (a person already in view would otherwise be learned as background; the median seeding reduces, but does not remove, this risk).

## Calibrating for your mounting height

The area thresholds depend on camera height. After changing the mount, update `CAMERA_HEIGHT_CM` in `src/config.py`, then let the helper suggest values from live data:

```bash
python src/calibrate_area.py --mode background --seconds 8   # empty scene: noise floor
python src/calibrate_area.py --mode target     --seconds 8   # with a real target in view
```

It prints suggested constants and does not modify `config.py`; copy the values you want.

## Repository layout

| File | Purpose |
|---|---|
| `src/main.py` | Entry point: wires sensor, detector, transformer and tracker; prints telemetry; writes the dashboard |
| `src/hardware.py` | MLX90640 driver wrapper: I²C setup, dead-pixel table, bus-hang detection and recovery, simulation fallback |
| `src/processing.py` | Dead-pixel fix, double-exponential smoothing, dual-rate background, Sauvola threshold, morphology, contour merging |
| `src/spatial.py` | Pixel to floor-plane conversion (undistortion plus trigonometric projection) |
| `src/tracker.py` | Kalman tracks and Hungarian association; track lifecycle |
| `src/config.py` | All tunable constants |
| `src/calibrate_area.py` | Helper that suggests area-gate constants from live sensor statistics |

## Results and validation

- **Throughput:** about **7.8 FPS** live on a Raspberry Pi 5 with the full pipeline. The sensor itself refreshes at 16 Hz; per-frame time is dominated by I²C transactions rather than by the processing.
- **Background stability (simulation):** a stationary 4×4-pixel target at +2.2 °C over a 22 °C field with 0.5 °C noise, run for 3000 frames (about 187 s). With a per-frame classification rule the target is absorbed into the background and is never detected again after about 19 s. With the hysteresis rule used here, detection stays between 88% and 100% for the whole run.
- **Calibration lesson:** using a reference height of 10 cm on a rig mounted at 75 cm collapsed the area gate and split one person into several tracks. The fix was to use the actual mounting height as the reference and raise the absolute area floor (see `src/config.py` and `src/calibrate_area.py`).

The evidence above is simulation and bench observation. There is no ground-truth tracking-accuracy benchmark yet.

## Known limitations

- Temperatures are the sensor's own readings (the hottest pixel inside a target's blob relative to the background). No emissivity, distance or ambient correction is applied, so treat them as apparent temperatures, not calibrated absolute ones.
- The camera matrix and distortion coefficients in `config.py` are assumed values, not the result of a calibration procedure.
- The ground projection assumes targets lie near a flat floor plane at the configured height.
- At 32×24 resolution, targets that are close together or partly out of frame can merge or be clipped; the edge margin and merge gap are tunable.
- The dashboard preview is not lens-corrected; only the tracked coordinates are.
- No automated tests yet.

## Roadmap

- Ground-truth evaluation (tape-measured positions, ID-switch counts).
- Tests for the tracker and the background model.
- A `requirements.txt` with pinned versions.
- Per-stage timing and CPU/power measurements on the Pi.


## License

Released under the [MIT License](LICENSE).
