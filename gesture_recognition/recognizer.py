from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Deque, Tuple
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class GestureEvent:
	type: str
	handedness: str
	confidence: float
	timestamp_ms: int
	features: Optional[Dict[str, float]] = None


class GestureRecognizer:
	def __init__(
		self,
		min_detection_confidence: float = 0.5,
		min_tracking_confidence: float = 0.5,
		pinch_enter: float = 0.35,
		pinch_exit: float = 0.45,
		smoothing_alpha: float = 0.6,
		max_num_hands: int = 2,
	):
		self._callback: Optional[Callable[[GestureEvent], None]] = None
		self._hands = mp.solutions.hands.Hands(
			static_image_mode=False,
			max_num_hands=max_num_hands,
			model_complexity=1,
			min_detection_confidence=min_detection_confidence,
			min_tracking_confidence=min_tracking_confidence,
		)
		self._drawing = mp.solutions.drawing_utils
		self._drawing_styles = mp.solutions.drawing_styles
		self._running = False
		self._debug = False
		self._pinch_enter = pinch_enter
		self._pinch_exit = pinch_exit
		self._alpha = smoothing_alpha
		self._last_confidence: Dict[str, float] = {}
		self._cooldown_ms = 250
		self._last_emit_ms: Dict[str, int] = {}
		self._enabled_gestures: Optional[set[str]] = None  # None means all
		# Overlay label smoothing (separate from event debounce)
		self._last_frame_labels: List[str] = []
		self._last_nonempty_labels: List[str] = []
		self._last_nonempty_time_ms: int = 0
		self._label_hold_ms: int = 400
		# Dynamic gesture state
		self._motion_history: Dict[str, Deque[Tuple[float, float, int]]] = {"Left": deque(), "Right": deque()}
		self._swipe_window_size: int = 10
		self._swipe_speed_thresh: float = 0.6  # normalized units per second (easier to trigger)
		self._swipe_min_cosine: float = 0.75
		self._swipe_min_displacement: float = 0.08
		self._pinch_dwell_ms: int = 250
		self._pinch_active: Dict[str, bool] = {"Left": False, "Right": False}
		self._pinch_start_ms: Dict[str, int] = {"Left": 0, "Right": 0}
		self._pinch_hold_emitted: Dict[str, bool] = {"Left": False, "Right": False}
		# Debug metrics
		self._debug_swipe_metrics: Dict[str, Dict[str, float]] = {"Left": {}, "Right": {}}

	def on_event(self, callback: Callable[[GestureEvent], None]) -> None:
		self._callback = callback

	def start(self, camera_index: int = 0, debug: bool = False) -> None:
		self._debug = debug
		cap = cv2.VideoCapture(camera_index)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		self._running = True
		prev_t = time.time()
		while self._running:
			ok, frame = cap.read()
			if not ok:
				break
			# Mirror for user-friendly UX
			frame = cv2.flip(frame, 1)
			events = self.process_frame(frame)
			for ev in events:
				if self._callback:
					self._callback(ev)
			if self._debug:
				# Overlay current gestures (from per-frame classification, not debounced events) and FPS
				elapsed = time.time() - prev_t
				fps = 1.0 / elapsed if elapsed > 0 else 0.0
				prev_t = time.time()
				now_ms = int(time.time() * 1000)
				if self._last_frame_labels:
					label = ", ".join(self._last_frame_labels)
				elif (now_ms - self._last_nonempty_time_ms) < self._label_hold_ms:
					label = ", ".join(self._last_nonempty_labels)
				else:
					label = "(none)"
				cv2.putText(frame, f"gestures: {label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
				# Draw swipe trails and metrics per hand
				for hand_id in ("Left", "Right"):
					buf = self._motion_history.get(hand_id)
					if buf and len(buf) >= 2:
						pts = np.array([[int(x * frame.shape[1]), int(y * frame.shape[0])] for x, y, _ in buf], dtype=np.int32)
						cv2.polylines(frame, [pts], False, (0, 200, 255), 2)
						m = self._debug_swipe_metrics.get(hand_id, {})
						if m:
							text = f"{hand_id[:1]} dv={m.get('disp',0):.2f} spd={m.get('speed',0):.2f} cos={m.get('cos',0):.2f}"
							cv2.putText(frame, text, (12, 84 if hand_id=="Left" else 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 0), 2)
				cv2.putText(frame, f"FPS: {fps:.1f} | q: quit", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
				cv2.imshow("Gesture Recognition", frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					self.stop()
		cap.release()
		cv2.destroyAllWindows()

	def stop(self) -> None:
		self._running = False

	def process_frame(self, frame_bgr: np.ndarray) -> List[GestureEvent]:
		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		results = self._hands.process(frame_rgb)
		events: List[GestureEvent] = []
		if not results.multi_hand_landmarks:
			return events

		frame_types: set[str] = set()
		for hand_landmarks, handedness in zip(
			results.multi_hand_landmarks,
			results.multi_handedness,
		):
			label = handedness.classification[0].label  # "Left" or "Right"
			hand_score = handedness.classification[0].score
			lm = hand_landmarks.landmark
			palm_scale = self._estimate_palm_scale(lm)
			finger_states = self._finger_states(lm, palm_scale)
			static_events = self._classify_static(finger_states, lm, palm_scale, label, hand_score)
			for ev in static_events:
				if self._is_enabled(ev.type) and self._should_emit(ev.type):
					events.append(self._smooth(ev))
				frame_types.add(ev.type)
			# Dynamic updates per hand
			self._update_motion_history(label, lm)
			dyn_events = self._detect_dynamic(label, lm, palm_scale)
			for ev in dyn_events:
				if self._is_enabled(ev.type) and self._should_emit(ev.type):
					events.append(self._smooth(ev))
					frame_types.add(ev.type)
			if self._debug:
				self._drawing.draw_landmarks(
					frame_bgr,
					hand_landmarks,
					mp.solutions.hands.HAND_CONNECTIONS,
					self._drawing_styles.get_default_hand_landmarks_style(),
					self._drawing_styles.get_default_hand_connections_style(),
				)
		# Update overlay label cache
		now_ms = int(time.time() * 1000)
		if frame_types:
			self._last_frame_labels = sorted(frame_types)
			self._last_nonempty_labels = self._last_frame_labels
			self._last_nonempty_time_ms = now_ms
		else:
			self._last_frame_labels = []
		return events

	def _centroid(self, lm) -> Tuple[float, float]:
		w = np.array([lm[0].x, lm[0].y])
		i = np.array([lm[5].x, lm[5].y])
		p = np.array([lm[17].x, lm[17].y])
		c = (w + i + p) / 3.0
		return float(c[0]), float(c[1])

	def _update_motion_history(self, handedness: str, lm) -> None:
		x, y = self._centroid(lm)
		t = int(time.time() * 1000)
		buf = self._motion_history.setdefault(handedness, deque(maxlen=self._swipe_window_size))
		# Ensure maxlen is set
		if buf.maxlen != self._swipe_window_size:
			self._motion_history[handedness] = deque(buf, maxlen=self._swipe_window_size)
			buf = self._motion_history[handedness]
		buf.append((x, y, t))

	def _detect_dynamic(self, handedness: str, lm, palm_scale: float) -> List[GestureEvent]:
		events: List[GestureEvent] = []
		# Swipes
		swipe = self._detect_swipe(handedness)
		if swipe:
			events.append(GestureEvent(swipe, handedness, 0.9, int(time.time() * 1000)))
			# After a swipe, clear history to avoid repeated triggers
			self._motion_history[handedness].clear()
		# Pinch hold/release
		ph_events = self._detect_pinch_hold_release(handedness, lm, palm_scale)
		events.extend(ph_events)
		return events

	def _detect_swipe(self, handedness: str) -> Optional[str]:
		buf = self._motion_history.get(handedness)
		if not buf or len(buf) < max(3, self._swipe_window_size // 2):
			return None
		x0, y0, t0 = buf[0]
		x1, y1, t1 = buf[-1]
		dt = max(1, t1 - t0) / 1000.0
		dx, dy = (x1 - x0), (y1 - y0)
		disp = np.hypot(dx, dy)
		vx, vy = dx / dt, dy / dt
		speed = np.hypot(vx, vy)
		# Store debug metrics early
		self._debug_swipe_metrics[handedness] = {"disp": float(disp), "speed": float(speed)}
		if disp < self._swipe_min_displacement or speed < self._swipe_speed_thresh:
			return None
		# Direction stability
		dir_vec = np.array([dx, dy])
		norm = np.linalg.norm(dir_vec)
		if norm <= 1e-6:
			return None
		dir_unit = dir_vec / norm
		cosines: List[float] = []
		for (xa, ya, _), (xb, yb, _) in zip(list(buf)[:-1], list(buf)[1:]):
			seg = np.array([xb - xa, yb - ya])
			seg_norm = np.linalg.norm(seg)
			if seg_norm > 1e-6:
				cos = float(np.dot(seg / seg_norm, dir_unit))
				cosines.append(cos)
		mean_cos = (sum(cosines) / len(cosines)) if cosines else 0.0
		self._debug_swipe_metrics[handedness]["cos"] = float(mean_cos)
		if not cosines or mean_cos < self._swipe_min_cosine:
			return None
		# Classify cardinal direction by dominant axis
		if abs(dx) >= abs(dy):
			return "swipe_right" if dx > 0 else "swipe_left"
		else:
			return "swipe_down" if dy > 0 else "swipe_up"

	def _detect_pinch_hold_release(self, handedness: str, lm, palm_scale: float) -> List[GestureEvent]:
		events: List[GestureEvent] = []
		now_ms = int(time.time() * 1000)
		thumb_tip = np.array([lm[4].x, lm[4].y])
		index_tip = np.array([lm[8].x, lm[8].y])
		pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
		pinch_on = pinch_dist < self._pinch_enter * palm_scale
		pinch_off = pinch_dist > self._pinch_exit * palm_scale
		active = self._pinch_active[handedness]
		# Transition on
		if pinch_on and not active:
			self._pinch_active[handedness] = True
			self._pinch_start_ms[handedness] = now_ms
			self._pinch_hold_emitted[handedness] = False
		# While active, check dwell
		if self._pinch_active[handedness] and not self._pinch_hold_emitted[handedness]:
			if (now_ms - self._pinch_start_ms[handedness]) >= self._pinch_dwell_ms:
				# Emit pinch_hold once
				events.append(GestureEvent("pinch_hold", handedness, 0.9, now_ms, {"pinch_dist": pinch_dist}))
				self._pinch_hold_emitted[handedness] = True
		# Transition off
		if pinch_off and active:
			self._pinch_active[handedness] = False
			# Emit pinch_release if we had held long enough (or always emit)
			if self._pinch_hold_emitted[handedness]:
				events.append(GestureEvent("pinch_release", handedness, 0.9, now_ms))
			self._pinch_hold_emitted[handedness] = False
			self._pinch_start_ms[handedness] = 0
		return events

	def _estimate_palm_scale(self, lm) -> float:
		wrist = np.array([lm[0].x, lm[0].y])
		index_mcp = np.array([lm[5].x, lm[5].y])
		middle_mcp = np.array([lm[9].x, lm[9].y])
		pinky_mcp = np.array([lm[17].x, lm[17].y])
		span = np.linalg.norm(index_mcp - pinky_mcp)
		depth = np.linalg.norm(wrist - middle_mcp)
		return max(1e-6, (span + depth) * 0.5)

	def _finger_states(self, lm, palm_scale: float) -> Dict[str, bool]:
		# Landmarks indices: https://google.github.io/mediapipe/solutions/hands.html
		# Heuristic: tip higher (smaller y) than pip and sufficiently far from wrist implies extended
		wrist = np.array([lm[0].x, lm[0].y])
		def extended(tip_idx: int, pip_idx: int) -> bool:
			return (lm[tip_idx].y < lm[pip_idx].y) and (
				np.linalg.norm(np.array([lm[tip_idx].x, lm[tip_idx].y]) - wrist) > 0.6 * palm_scale
			)
		states = {
			"thumb": lm[4].x < lm[3].x if True else False,  # placeholder lateral check in mirrored space
			"index": extended(8, 6),
			"middle": extended(12, 10),
			"ring": extended(16, 14),
			"pinky": extended(20, 18),
		}
		return states

	def _classify_static(
		self,
		finger_states: Dict[str, bool],
		lm,
		palm_scale: float,
		handedness: str,
		hand_score: float,
	) -> List[GestureEvent]:
		def now_ms() -> int:
			return int(time.time() * 1000)
		thumb_tip = np.array([lm[4].x, lm[4].y])
		index_tip = np.array([lm[8].x, lm[8].y])
		pinch_dist = np.linalg.norm(thumb_tip - index_tip)

		candidates: List[GestureEvent] = []

		# open_palm
		if all(finger_states.values()):
			conf = self._score(hand_score, margin=0.5)
			candidates.append(GestureEvent("open_palm", handedness, conf, now_ms()))

		# fist
		if not any(finger_states.values()):
			conf = self._score(hand_score, margin=0.5)
			candidates.append(GestureEvent("fist", handedness, conf, now_ms()))

		# point
		if finger_states["index"] and not (finger_states["middle"] or finger_states["ring"] or finger_states["pinky"]):
			conf = self._score(hand_score, margin=0.5)
			candidates.append(GestureEvent("point", handedness, conf, now_ms()))

		# pinch with hysteresis
		pinch_on = pinch_dist < self._pinch_enter * palm_scale
		pinch_off = pinch_dist > self._pinch_exit * palm_scale
		pinch_state = self._last_confidence.get(f"pinch_{handedness}", 0.0) > 0.5
		if pinch_on or (pinch_state and not pinch_off):
			margin = max(0.0, (self._pinch_exit * palm_scale - pinch_dist) / (self._pinch_exit * palm_scale))
			conf = self._score(hand_score, margin=margin)
			candidates.append(GestureEvent("pinch", handedness, conf, now_ms(), {"pinch_dist": float(pinch_dist)}))

		return candidates

	def _score(self, hand_score: float, margin: float) -> float:
		return float(np.clip(0.5 * hand_score + 0.5 * margin, 0.0, 1.0))

	def _smooth(self, event: GestureEvent) -> GestureEvent:
		key = f"{event.type}_{event.handedness}"
		prev = self._last_confidence.get(key, 0.0)
		smoothed = self._alpha * event.confidence + (1.0 - self._alpha) * prev
		self._last_confidence[key] = smoothed
		return GestureEvent(event.type, event.handedness, smoothed, event.timestamp_ms, event.features)

	def _should_emit(self, gesture_type: str) -> bool:
		key = f"{gesture_type}"
		now_ms = int(time.time() * 1000)
		last = self._last_emit_ms.get(key, 0)
		if now_ms - last < self._cooldown_ms:
			return False
		self._last_emit_ms[key] = now_ms
		return True

	def _is_enabled(self, gesture_type: str) -> bool:
		if self._enabled_gestures is None:
			return True
		return gesture_type in self._enabled_gestures

	def load_config(self, path: str) -> None:
		import json
		with open(path, "r", encoding="utf-8") as f:
			cfg = json.load(f)
		# Thresholds
		self._pinch_enter = float(cfg.get("pinch_enter", self._pinch_enter))
		self._pinch_exit = float(cfg.get("pinch_exit", self._pinch_exit))
		self._alpha = float(cfg.get("smoothing_alpha", self._alpha))
		self._cooldown_ms = int(cfg.get("cooldown_ms", self._cooldown_ms))
		# Dynamic params
		self._swipe_window_size = int(cfg.get("swipe_window_size", self._swipe_window_size))
		self._swipe_speed_thresh = float(cfg.get("swipe_speed_thresh", self._swipe_speed_thresh))
		self._swipe_min_cosine = float(cfg.get("swipe_min_cosine", self._swipe_min_cosine))
		self._swipe_min_displacement = float(cfg.get("swipe_min_displacement", self._swipe_min_displacement))
		self._pinch_dwell_ms = int(cfg.get("pinch_dwell_ms", self._pinch_dwell_ms))
		# Enabled gestures
		enabled = cfg.get("enabled_gestures")
		if isinstance(enabled, list):
			self._enabled_gestures = set(map(str, enabled))
		# Mediapipe options cannot be changed after creation (simple approach): advise recreate if changed
		mdc = cfg.get("min_detection_confidence")
		mtc = cfg.get("min_tracking_confidence")
		mnh = cfg.get("max_num_hands")
		if any(v is not None for v in (mdc, mtc, mnh)):
			print("Note: min_* confidences and max_num_hands require re-instantiation to take effect.")


