import json
from pathlib import Path
import asyncio
import threading
from gesture_recognition import GestureRecognizer, GestureEvent
from browser_control.controller import BrowserController
from browser_control.router import ActionRouter


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
    router = ActionRouter(controller, mapping_path="gesture_actions.json")

    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        controller.run_in_loop(loop)
        loop.run_forever()

    threading.Thread(target=run_loop, daemon=True).start()
    router.attach_loop(loop)

    recognizer.on_event(router.handle_event)
    recognizer.start(camera_index=0, debug=True)


if __name__ == "__main__":
    main()
