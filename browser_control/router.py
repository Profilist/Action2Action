from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, Any

from gesture_recognition import GestureEvent
from .controller import BrowserController, Action


class ActionRouter:
    def __init__(
        self, controller: BrowserController, mapping_path: str = "gesture_actions.json"
    ) -> None:
        self._controller = controller
        with open(mapping_path, "r", encoding="utf-8") as f:
            self._cfg: Dict[str, Any] = json.load(f)
        self._min_conf: float = float(self._cfg.get("min_confidence", 0.7))
        self._cooldowns: Dict[str, int] = {
            k: int(v) for k, v in self._cfg.get("cooldowns_ms", {}).items()
        }
        self._last_emit_ms: Dict[str, int] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _cooldown_ok(self, key: str) -> bool:
        now = int(time.time() * 1000)
        cd = self._cooldowns.get(key, 600)
        if now - self._last_emit_ms.get(key, 0) < cd:
            return False
        self._last_emit_ms[key] = now
        return True

    def handle_event(self, ev: GestureEvent) -> None:
        if ev.confidence < self._min_conf:
            return
        if ev.type == "fist":
            # future: implement queue clearing/pause
            return
        key_specific = f"{ev.type}:{ev.handedness}"
        action_def = self._cfg.get("actions", {}).get(key_specific) or self._cfg.get(
            "actions", {}
        ).get(ev.type)
        if not action_def:
            return
        if not self._cooldown_ok(ev.type):
            return
        task = str(action_def["task"])
        if self._loop is None:
            # No loop attached; drop the action
            return
        asyncio.run_coroutine_threadsafe(
            self._controller.enqueue(Action(name=ev.type, task=task)),
            self._loop,
        )
