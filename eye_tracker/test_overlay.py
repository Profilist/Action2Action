from __future__ import annotations

import cv2
from overlay import EyeGazeOverlay


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    overlay = EyeGazeOverlay(
        box_size=64,
        box_color=(0, 0, 255),
        box_thickness=3,
        smoothing_alpha=0.8,
        deadzone_px=20,
        invert_x=False,   # flip if camera is mirrored
        invert_y=False,  # set True if vertical feels inverted
        # Calibrated from your corner measurements:
        # nx_left≈(0.517+0.541)/2=0.529, nx_right≈(0.766+0.750)/2=0.758
        # ny_top≈(0.955+0.727)/2=0.841, ny_bottom≈(1.056+1.066)/2=1.061
        # x_scale=1/(0.758-0.529)=4.3668, x_offset=-0.529*x_scale≈-2.3100
        # y_scale=1/(1.061-0.841)=4.5455, y_offset=-0.841*y_scale≈-3.8227
        x_scale=4.3668,
        x_offset=-2.3100,
        y_scale=4.5455,
        y_offset=-3.8227,
        grid_px=32,
        max_step_px=20,
    )
    overlay.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # If your camera feed is mirrored by default, keep this flip.
            # If you disable it, adjust invert_x accordingly.
            frame = cv2.flip(frame, 1)
            overlay.update_from_frame(frame)
            # Annotate like example.py
            frame_annot = overlay.annotate_frame(frame)
            cv2.imshow("Overlay Test (press q to quit)", frame_annot)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        overlay.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


