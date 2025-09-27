import json
from pathlib import Path
import cv2
from gesture_recognition import GestureRecognizer, GestureEvent


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
	recognizer.on_event(print_event)
	recognizer.start(camera_index=0, debug=True)


if __name__ == "__main__":
	main()


