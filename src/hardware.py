import time
import numpy as np
import config

try:
    import board
    import busio
    import adafruit_mlx90640
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False


class ThermalSensor:
    """
    Physical bus I/O interface for the MLX90640.

    Step 1 optimization: the I2C clock is elevated to 400kHz (config.I2C_FREQUENCY_HZ) to
    reduce transaction delays and clock stretching. A frame-redundancy check detects bus
    hangs (the driver silently returning the same frame on consecutive reads) and recovers
    by re-probing the sensor's hardware address and resetting the bus, instead of requiring
    a full process restart.
    """

    def __init__(self):
        if HARDWARE_AVAILABLE:
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=config.I2C_FREQUENCY_HZ)
            self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
            self.frame = [0] * 768
            self._last_frame = None
            self._repeat_count = 0
            self._load_dead_pixel_table()
            print(f"[INFO] Connected to live MLX90640 sensor hardware over I2C @ {config.I2C_FREQUENCY_HZ // 1000}kHz.")
        else:
            print("[WARN] Hardware interface missing. Booting script in simulation environment.")
            self.frame = None

    def _load_dead_pixel_table(self):
        """Pulls the onboard EEPROM-flagged bad-pixel coordinates so processing.py can
        correct them before any spatial interpolation (Step 2)."""
        try:
            bad_pixels = getattr(self.mlx, "broken_pixels", None) or []
            outlier_pixels = getattr(self.mlx, "outlier_pixels", None) or []
            coords = []
            for idx in list(bad_pixels) + list(outlier_pixels):
                if idx is None:
                    continue
                row, col = divmod(int(idx), config.RAW_RES[0])
                coords.append((row, col))
            config.KNOWN_DEAD_PIXELS = coords
            if coords:
                print(f"[INFO] Loaded {len(coords)} EEPROM-flagged dead/outlier pixel(s).")
        except Exception:
            # Library version doesn't expose the calibration table -- the runtime
            # NaN/out-of-range outlier detector in processing.py still catches these.
            config.KNOWN_DEAD_PIXELS = []

    def _recover_bus_hang(self):
        """Re-probes the sensor's I2C address and resets the bus without a full reboot."""
        try:
            print("[WARN] I2C frame hang detected (identical consecutive frames). Resetting bus...")
            while not self.i2c.try_lock():
                pass
            self.i2c.scan()  # Active probe -- forces a transaction on MLX90640_I2C_ADDRESS
            self.i2c.unlock()
            self._repeat_count = 0
        except Exception:
            pass

    def get_raw_matrix(self):
        if HARDWARE_AVAILABLE:
            try:
                self.mlx.getFrame(self.frame)
                matrix = np.array(self.frame, dtype=np.float32).reshape(
                    (config.RAW_RES[1], config.RAW_RES[0])
                )

                # Frame-redundancy hang check: identical consecutive frames indicate the
                # I2C transaction has stalled rather than genuinely static ambient temps.
                if self._last_frame is not None and np.array_equal(matrix, self._last_frame):
                    self._repeat_count += 1
                    if self._repeat_count >= config.I2C_HANG_REPEAT_LIMIT:
                        self._recover_bus_hang()
                        return None
                else:
                    self._repeat_count = 0

                self._last_frame = matrix
                return matrix
            except Exception:
                # Catch temporary hardware dropouts without interrupting code execution loops
                return None
        else:
            time.sleep(0.0625)  # Mimic native 16Hz hardware timing clock
            w, h = config.RAW_RES
            matrix = np.full((h, w), 24.5 + 0.1 * np.sin(time.time() * 0.05), dtype=np.float32)

            # Synthesize an agnostic hot profile and cold profile to verify processing logic
            t = time.time() * 0.3
            cx1, cy1 = int(w * 0.5 + 6 * np.cos(t)), int(h * 0.5 + 4 * np.sin(t))
            cx2, cy2 = int(w * 0.31 + 3 * np.sin(t * 1.5)), int(h * 0.75 + 2 * np.cos(t * 1.2))

            cy1c, cx1c = np.clip(cy1, 1, h - 2), np.clip(cx1, 2, w - 2)
            cy2c, cx2c = np.clip(cy2, 0, h - 1), np.clip(cx2, 0, w - 1)

            matrix[cy1c - 1:cy1c + 2, cx1c - 2:cx1c + 2] = 34.0  # Hot Target Signatures
            matrix[cy2c, cx2c] = 16.5                            # Cold Target Signatures
            return matrix
