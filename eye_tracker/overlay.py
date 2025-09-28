from __future__ import annotations

import threading
import queue
import sys
from pathlib import Path
from typing import Optional, Tuple
import math
from threading import Lock

# OpenCV is not strictly required in this module; frames are passed through


class EyeGazeOverlay:
    """
    Always-on-top transparent overlay that draws a small box at the current
    gaze position. It computes gaze using the bundled GazeTracking library
    by receiving frames via update_from_frame().

    Usage:
        overlay = EyeGazeOverlay()
        overlay.start()
        ... per frame: overlay.update_from_frame(frame_bgr)
        overlay.stop()
    """

    def __init__(
        self,
        box_size: int = 50,
        box_color: Tuple[int, int, int] = (255, 0, 0),  # BGR
        box_thickness: int = 3,
        smoothing_alpha: float = 0.35,
        deadzone_px: int = 8,
        invert_x: bool = True,
        invert_y: bool = False,
        x_scale: float = 1.0,
        x_offset: float = 0.0,
        y_scale: float = 1.0,
        y_offset: float = 0.0,
        grid_px: int = 12,
        max_step_px: int = 40,
    ) -> None:
        self._box_size = int(box_size)
        self._box_color_bgr = tuple(int(c) for c in box_color)
        self._box_thickness = int(box_thickness)
        self._alpha = float(max(0.0, min(1.0, smoothing_alpha)))
        self._deadzone_px = int(max(0, deadzone_px))
        self._invert_x = bool(invert_x)
        self._invert_y = bool(invert_y)
        self._x_scale = float(x_scale)
        self._x_offset = float(x_offset)
        self._y_scale = float(y_scale)
        self._y_offset = float(y_offset)
        self._grid_px = int(max(0, grid_px))
        self._max_step_px = int(max(1, max_step_px))

        self._root = None  # type: ignore[var-annotated]
        self._canvas = None  # type: ignore[var-annotated]
        self._screen_w = 0
        self._screen_h = 0
        self._coords_queue: "queue.Queue[Tuple[int, int]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Lazy-load gaze tracking after adding its path
        self._gaze = None
        # Smoothed screen coordinates
        self._ema_x: Optional[float] = None
        self._ema_y: Optional[float] = None
        # Last info for UI/debug
        self._last_left_pupil: Optional[Tuple[int, int]] = None
        self._last_right_pupil: Optional[Tuple[int, int]] = None
        self._last_screen_xy: Optional[Tuple[int, int]] = None
        self._last_status: str = ""
        # Last normalized gaze (after inversion, before scale/offset)
        self._last_nx: Optional[float] = None
        self._last_ny: Optional[float] = None
        # State lock for cross-thread reads
        self._state_lock: Lock = Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # Close Tk root if present
        try:
            if self._root is not None:
                self._root.quit()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def update_from_frame(self, frame_bgr) -> None:
        """
        Accept a BGR frame (OpenCV) and update the overlay position
        according to the estimated gaze ratios.
        """
        try:
            if self._gaze is None:
                self._init_gaze()
            self._gaze.refresh(frame_bgr)
            # Compute ratios if pupils are available
            if self._gaze.pupils_located:
                h_ratio = self._gaze.horizontal_ratio()  # 0.0 right .. 1.0 left
                v_ratio = self._gaze.vertical_ratio()  # 0.0 top .. 1.0 bottom
                if h_ratio is None or v_ratio is None:
                    return
                # Save pupil coords
                self._last_left_pupil = self._gaze.pupil_left_coords()
                self._last_right_pupil = self._gaze.pupil_right_coords()
                # Save status similar to example.py
                if self._gaze.is_blinking():
                    self._last_status = "Blinking"
                elif self._gaze.is_right():
                    self._last_status = "Looking right"
                elif self._gaze.is_left():
                    self._last_status = "Looking left"
                elif self._gaze.is_center():
                    self._last_status = "Looking center"
                else:
                    self._last_status = ""
                # Map to screen coordinates. Tk coords start at top-left (0,0).
                # horizontal_ratio(): 0.0 is rightmost, so mirror for x.
                if self._screen_w <= 0 or self._screen_h <= 0:
                    # If overlay not yet initialized, skip until it is
                    return
                # Normalize to [0,1] with optional inversion
                nx = 1.0 - float(h_ratio) if self._invert_x else float(h_ratio)
                ny = 1.0 - float(v_ratio) if self._invert_y else float(v_ratio)
                # Persist normalized, then apply scale/offset
                self._last_nx, self._last_ny = nx, ny
                # Apply scale/offset, then map to pixels
                target_x = (self._x_offset + self._x_scale * nx) * self._screen_w
                target_y = (self._y_offset + self._y_scale * ny) * self._screen_h

                # Initialize EMA if first sample
                if self._ema_x is None or self._ema_y is None:
                    self._ema_x, self._ema_y = target_x, target_y
                # Dead-zone: ignore tiny movements
                else:
                    dx = target_x - self._ema_x
                    dy = target_y - self._ema_y
                    if math.hypot(dx, dy) < self._deadzone_px:
                        target_x, target_y = self._ema_x, self._ema_y
                # EMA smoothing
                self._ema_x = self._alpha * target_x + (1.0 - self._alpha) * self._ema_x
                self._ema_y = self._alpha * target_y + (1.0 - self._alpha) * self._ema_y

                # Optional grid snap
                x_snap = self._ema_x
                y_snap = self._ema_y
                if self._grid_px > 0:
                    x_snap = round(x_snap / self._grid_px) * self._grid_px
                    y_snap = round(y_snap / self._grid_px) * self._grid_px

                # Step limit relative to last emitted point
                if self._last_screen_xy is not None:
                    lx, ly = self._last_screen_xy
                    dx = x_snap - lx
                    dy = y_snap - ly
                    dist = math.hypot(dx, dy)
                    if dist > self._max_step_px and dist > 1e-6:
                        scale = self._max_step_px / dist
                        x_snap = lx + dx * scale
                        y_snap = ly + dy * scale

                # Clamp and enqueue integer coords
                x_i = max(0, min(self._screen_w - 1, int(x_snap)))
                y_i = max(0, min(self._screen_h - 1, int(y_snap)))
                with self._state_lock:
                    self._last_screen_xy = (x_i, y_i)
                if self._running:
                    try:
                        self._coords_queue.put_nowait((x_i, y_i))
                    except queue.Full:
                        pass
        except Exception:
            # Swallow errors from optional gaze overlay to not disturb main pipeline
            pass

    # ---- internals ----

    def _init_gaze(self) -> None:
        # Ensure the bundled GazeTracking package is importable
        here = Path(__file__).resolve().parent
        gaze_pkg = here / "GazeTracking"
        spath = str(gaze_pkg)
        if spath not in sys.path:
            sys.path.insert(0, spath)
        from gaze_tracking import GazeTracking  # type: ignore

        self._gaze = GazeTracking()

    # Public API for test/annotation
    def get_last_info(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[Tuple[int, int]], str]:
        """Return (left_pupil, right_pupil, screen_xy, status)."""
        with self._state_lock:
            screen_xy = self._last_screen_xy
            left = self._last_left_pupil
            right = self._last_right_pupil
            status = self._last_status
        return left, right, screen_xy, status

    def get_last_screen_xy(self) -> Optional[Tuple[int, int]]:
        """Return latest screen coordinates for gaze overlay, if available."""
        with self._state_lock:
            return self._last_screen_xy

    def annotate_frame(self, frame_bgr):
        """Return frame annotated with pupil crosshairs and text similar to example.py."""
        try:
            if self._gaze is None:
                return frame_bgr
            # Use underlying GazeTracking's annotated_frame for crosshairs
            frame = self._gaze.annotated_frame()
            import cv2  # local import to avoid module-level dependency
            text = self._last_status or ""
            if text:
                cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (147, 58, 31), 2)
            # Pupil coordinates
            left = self._last_left_pupil
            right = self._last_right_pupil
            cv2.putText(frame, f"Left pupil:  {left}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 1)
            cv2.putText(frame, f"Right pupil: {right}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 1)
            # Screen coordinates (overlay box)
            screen_xy = self._last_screen_xy
            if screen_xy is not None:
                cv2.putText(frame, f"Screen: {screen_xy}", (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.8, (80, 120, 200), 1)
            # Normalized values
            if self._last_nx is not None and self._last_ny is not None:
                cv2.putText(
                    frame,
                    f"nx, ny: ({self._last_nx:.3f}, {self._last_ny:.3f})",
                    (90, 235),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (200, 200, 80),
                    1,
                )
            return frame
        except Exception:
            return frame_bgr

    def _run_tk(self) -> None:
        try:
            import tkinter as tk
        except Exception:
            # Tkinter not available; disable overlay
            self._running = False
            return

        self._root = tk.Tk()
        self._root.title("EyeGazeOverlay")
        # Fullscreen, no border, always on top
        self._root.overrideredirect(True)
        self._root.wm_attributes("-topmost", 1)

        # Transparent background using a chroma key color
        transparent = "magenta"
        self._root.configure(bg=transparent)
        try:
            # Windows supports -transparentcolor
            self._root.wm_attributes("-transparentcolor", transparent)
        except Exception:
            # Fallback: set low alpha for the window
            try:
                self._root.wm_attributes("-alpha", 0.3)
            except Exception:
                pass

        # Determine screen size
        self._screen_w = int(self._root.winfo_screenwidth())
        self._screen_h = int(self._root.winfo_screenheight())
        self._root.geometry(f"{self._screen_w}x{self._screen_h}+0+0")

        self._canvas = tk.Canvas(
            self._root,
            width=self._screen_w,
            height=self._screen_h,
            highlightthickness=0,
            bg=transparent,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Create initial rectangle off-screen
        self._rect = self._canvas.create_rectangle(
            -100, -100, -50, -50,
            outline=self._rgb_hex(self._box_color_bgr[::-1]),  # convert BGR to RGB
            width=self._box_thickness,
        )

        # Poll for queued coordinate updates
        def poll_queue() -> None:
            try:
                while True:
                    x, y = self._coords_queue.get_nowait()
                    self._move_rect(x, y)
            except queue.Empty:
                pass
            if self._running:
                self._root.after(16, poll_queue)

        self._root.after(16, poll_queue)
        try:
            self._root.mainloop()
        finally:
            try:
                self._root.destroy()
            except Exception:
                pass

    def _move_rect(self, x: int, y: int) -> None:
        if not self._canvas:
            return
        half = self._box_size // 2
        x0 = x - half
        y0 = y - half
        x1 = x + half
        y1 = y + half
        self._canvas.coords(self._rect, x0, y0, x1, y1)

    @staticmethod
    def _rgb_hex(rgb: Tuple[int, int, int]) -> str:
        r, g, b = rgb
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        return f"#{r:02x}{g:02x}{b:02x}"


