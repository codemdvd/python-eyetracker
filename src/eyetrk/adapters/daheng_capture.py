from __future__ import annotations

import sys
import cv2
from pathlib import Path

# gxipy lives at the project root as a local package — ensure it's on sys.path
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_GXIPY_OK = False
_GXIPY_ERR: str = ""
try:
    import gxipy as _gx  # type: ignore[import]
    _GXIPY_OK = True
except Exception as _e:
    _gx = None  # type: ignore[assignment]
    _GXIPY_ERR = f"{type(_e).__name__}: {_e}"


class DahengCapture:
    """
    cv2.VideoCapture-compatible wrapper for Daheng GxIpy industrial cameras.
    Converts MONO8 frames to BGR so MediaPipe adapters work without modification.
    """

    def __init__(
        self,
        device_index: int = 1,
        exposure_us: float = 5000.0,
        gain_db: float = 12.0,
    ) -> None:
        self._cam = None
        self._dm = None  # keep DeviceManager alive — GC would de-init the API
        self._ok = False
        self._width = 0
        self._height = 0

        if not _GXIPY_OK or _gx is None:
            print(f"[daheng] gxipy import failed — {_GXIPY_ERR}")
            return

        try:
            dm = _gx.DeviceManager()
            self._dm = dm
            dev_num, _ = dm.update_device_list()
            if dev_num == 0:
                print("[daheng] No devices found")
                return

            cam = dm.open_device_by_index(device_index)
            cam.TriggerMode.set(_gx.GxSwitchEntry.OFF)
            cam.ExposureAuto.set(_gx.GxAutoEntry.OFF)
            cam.GainAuto.set(_gx.GxAutoEntry.OFF)
            cam.PixelFormat.set(_gx.GxPixelFormatEntry.MONO8)
            cam.ExposureTime.set(float(exposure_us))
            cam.Gain.set(float(gain_db))

            # Read resolution from device registers — no frame grab needed
            try:
                self._width = int(cam.Width.get())
                self._height = int(cam.Height.get())
            except Exception:
                self._width = 0
                self._height = 0

            cam.stream_on()
            self._cam = cam
            self._ok = True
            print(
                f"[daheng] Camera opened (device_index={device_index}, "
                f"{self._width}x{self._height}, "
                f"exposure={exposure_us}µs, gain={gain_db}dB)"
            )
        except Exception as exc:
            print(f"[daheng] Failed to open camera: {exc}")
            self._cam = None

    def isOpened(self) -> bool:
        return self._ok

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def read(self) -> tuple[bool, object]:
        if not self._ok or self._cam is None:
            return False, None
        try:
            img = self._cam.data_stream[0].get_image(timeout=200)
            if img is None:
                return False, None
            # Drain any buffered frames — always deliver the freshest one
            for _ in range(16):
                try:
                    newer = self._cam.data_stream[0].get_image(timeout=1)
                    if newer is None:
                        break
                    img = newer
                except Exception:
                    break
            frame = img.get_numpy_array()
            if frame is None:
                return False, None
            # MONO8 (H, W) → BGR (H, W, 3) for MediaPipe / cv2 compatibility
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return True, bgr
        except Exception:
            return False, None

    def release(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stream_off()
                self._cam.close_device()
            except Exception:
                pass
            self._cam = None
        self._ok = False


def is_available() -> bool:
    return _GXIPY_OK
