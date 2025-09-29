# Action2Action (🥇 Winner of TechTo x Penseum Hackathon)
> Browser automation with shortcuts using gestures, eye tracking, and voice

<img width="1440" height="900" alt="maroon-interfaces-316779 framer app_(MacBook Air)" src="https://github.com/user-attachments/assets/88da4cb3-0be6-406b-91ff-e48dfdacd307" />



Combines real‑time gesture recognition, eye tracking, and voice recognition (soon!) with a Browser Use agent: 
- Map gestures like swipes, pinches, open palm, point, and fist to simple or complex actions, from opening a tab, switching slides, to extracting data or adding an event to your calendar through our [dashboard](https://a2-a-puce.vercel.app/)

### Key Features
- Real‑time gesture recognition via `gesture_recognition.GestureRecognizer` emitting structured `GestureEvent`s.
- Optional eye tracking overlay (`eye_tracker.EyeGazeOverlay`) with smoothing, deadzone, grid snap, and per‑screen calibration.
- Browser automation powered by Browser Use via `browser_control.BrowserController` and `ActionRouter`.
- JSON mapping (`gesture_actions.json`) to customize behavior of shortcuts (supports direct GUI interactions, AI tasks, and multi‑step sequences).
- Configurable thresholds and gesture enablement via `gesture_config.json`.

### Repository Structure
- `gesture_recognition/` — MediaPipe/OpenCV gesture pipeline
  - `recognizer.py` — core logic, events, static and dynamic gestures (swipes, pinch hold)
- `eye_tracker/` — gaze overlay utilities
  - `GazeTracking` — credits to [Antoine Lame's library](https://github.com/antoinelame/GazeTracking)
  - `overlay.py` — Tk overlay and gaze mapping; `GazeTracking` vendor code bundled
- `browser_control/` — browser agent integration
  - `controller.py` — queue‑based controller, direct command execution and AI agent tasks
  - `router.py` — routes `GestureEvent`s to actions based on JSON with cooldowns and optional context
- `demo.py` — end‑to‑end wiring: camera → recognizer → router → browser; runs eye overlay
- `gesture_actions.json` — gesture → action mapping
- `gesture_config.json` — gesture thresholds and options

---

## Requirements
- Python 3.10+
- OS: Windows (paths and hotkeys tuned for Chrome on Windows).
- Chrome installed. Close all Chrome windows before running so the agent can attach to your profile.

Python packages (see `requirements.txt`):
- mediapipe, opencv‑python, numpy
- browser‑use, pyautogui
- dlib (for bundled `GazeTracking`), setuptools

Note: `dlib` can be heavy to build. Prefer a Python environment where wheels are available.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Since `browser-use` requires OpenAI credentials, set environment variables per its documentation (e.g., `OPENAI_API_KEY`).

## Configuration
### Gesture recognition (`gesture_config.json`)
Controls thresholds, smoothing, cooldowns, enabled gestures, and dwell timings. Example keys:
- `pinch_enter`, `pinch_exit`, `smoothing_alpha`, `cooldown_ms`
- `enabled_gestures` — list of allowed gesture types
- Dynamic gestures: `swipe_window_size`, `swipe_speed_thresh`, `swipe_min_cosine`, `swipe_min_displacement`
- Dwell: `pinch_dwell_ms`, `static_dwell_ms`

`demo.py` auto‑loads this file if present.

### Gesture → Action mapping (`gesture_actions.json`)
Defines how gestures map to browser actions. Structure:
- `min_confidence`: minimum event confidence
- `cooldowns_ms`: per‑gesture cooldowns
- `actions`: mapping of keys to actions where key is `type` or `type:Hand` (e.g., `pinch_hold:Right`)

Each action supports one of:
- `command`: a built‑in direct command (e.g., `tab_open`, `tab_next`, `open_slides`, `next_slide`, `event_to_calendar`)
- `task`: a natural‑language instruction executed by the Browser‑Use agent
- `steps`: array of `{command,args?}` and/or `{task}` to run a sequence

Example excerpt:
```json
{
  "min_confidence": 0.3,
  "cooldowns_ms": { "swipe_left": 1200, "pinch_hold": 2000 },
  "actions": {
    "swipe_left": { "command": "nothing" },
    "pinch_hold:Right": { "command": "tab_open", "args": { "url": "https://example.com" } },
    "fist": { "steps": [
      {"command": "tab_open", "args": {"url": "https://docs.google.com/..."}},
      {"task": "Open the sheet and navigate to the patient tab, then stop."}
    ] }
  }
}
```

## Usage
### End‑to‑end demo (gestures → browser)
```bash
python demo.py
```
What it does:
- Starts eye‑gaze overlay (optional) and the Browser controller on a background event loop
- Runs the webcam recognizer with debug overlays (press `q` to quit)
- Routes recognized gestures to actions via `ActionRouter`

Tips:
- Ensure Chrome is closed before starting so the agent can attach to your default profile.
- If your camera feed is mirrored, `demo.py` flips frames for UX; adjust `invert_x`/`invert_y` in the overlay if needed.

## Core Concepts
### GestureEvent
Emitted by `GestureRecognizer` per frame when a gesture is detected:
- `type`: e.g., `open_palm`, `fist`, `point`, `pinch`, `pinch_hold`, `swipe_left/right/up/down`
- `handedness`: `Left` or `Right`
- `confidence`: smoothed confidence (0‑1)
- `timestamp_ms`: event time
- `features`: optional metrics (e.g., `pinch_dist`)

### Recognizer pipeline
- MediaPipe Hands for landmarks → rule‑based static gestures → dynamic gestures via short motion history (swipes) and dwell (pinch_hold)
- EMA smoothing and per‑gesture cooldown
- Optional config loaded from `gesture_config.json`

### Browser controller and router
- `BrowserController` maintains a persistent Chrome session and executes either direct commands (via `pyautogui` hotkeys and URL inputs) or AI tasks through `browser_use.Agent` + `ChatOpenAI`.
- `ActionRouter` enforces min confidence and cooldowns, enriches task steps with optional context (e.g., eye‑gaze screen coordinates), and enqueues actions onto the controller loop.

## Environment Variables (optional)
- `CHROME_PATH`, `CHROME_USER_DATA_DIR`, `CHROME_PROFILE` — override Chrome paths if needed.
- Any LLM keys required by `browser_use` (e.g., `OPENAI_API_KEY`).

## Troubleshooting
- Chrome doesn’t attach or commands do nothing:
  - Close all Chrome windows first. Verify `CHROME_*` paths if customized.
  - Some commands rely on window focus; the controller tries to focus Chrome by title.
- Webcam feed is mirrored/inverted:
  - Keep the horizontal flip in `demo.py`; adjust overlay `invert_x`/`invert_y` if gaze feels inverted.
- High false positives or jitter:
  - Increase `min_confidence` in `gesture_actions.json` and/or raise `static_dwell_ms` in `gesture_config.json`.
  - Increase swipe thresholds (`swipe_speed_thresh`, `swipe_min_displacement`).
- `dlib` install issues:
  - Use Python versions with prebuilt wheels; ensure Visual C++ Build Tools on Windows if compiling.
