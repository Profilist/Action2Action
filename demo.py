import json
from pathlib import Path
import asyncio
import threading
from gesture_recognition import GestureRecognizer, GestureEvent
from browser_control.controller import BrowserController
from browser_control.router import ActionRouter
from eye_tracker.overlay import EyeGazeOverlay


def print_event(event: GestureEvent) -> None:
    data = {
        "type": event.type,
        "handedness": event.handedness,
        "confidence": round(event.confidence, 3),
        "timestamp_ms": event.timestamp_ms,
        "features": event.features or {},
    }
    print(json.dumps(data))


def main() -> None:
    recognizer = GestureRecognizer()
    # Optional config
    cfg_path = Path("gesture_config.json")
    if cfg_path.exists():
        print(f"Loading config from {cfg_path}")
        recognizer.load_config(str(cfg_path))

    # Browser-Use wiring
    controller = BrowserController()
    overlay = EyeGazeOverlay(
        box_size=128,
        box_color=(0, 0, 255),
        box_thickness=3,
        smoothing_alpha=0.8,
        deadzone_px=20,
        invert_x=False,
        invert_y=False,
        x_scale=4.3668,
        x_offset=-2.3100,
        y_scale=4.5455,
        y_offset=-3.8227,
        grid_px=32,
        max_step_px=20,
    )

    def context_supplier() -> dict:
        xy = overlay.get_last_screen_xy()
        return {"screen_xy": xy} if xy is not None else {}

    router = ActionRouter(controller, mapping_path="gesture_actions.json", context_supplier=context_supplier)

    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        controller.run_in_loop(loop)
        loop.run_forever()

    threading.Thread(target=run_loop, daemon=True).start()
    router.attach_loop(loop)

    recognizer.on_event(router.handle_event)

    overlay.start()

    try:
        recognizer.start(camera_index=0, debug=True, frame_callback=overlay.update_from_frame)
    finally:
        overlay.stop()


if __name__ == "__main__":
    main()
