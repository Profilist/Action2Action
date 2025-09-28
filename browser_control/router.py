from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, Any, Optional, Callable

from gesture_recognition import GestureEvent
from .controller import BrowserController, Action


class ActionRouter:
    def __init__(
        self,
        controller: BrowserController,
        mapping_path: str = "gesture_actions.json",
        context_supplier: Optional[Callable[[], Dict[str, Any]]] = None,
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
        self._context_supplier = context_supplier

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
        key_specific = f"{ev.type}:{ev.handedness}"
        action_def = self._cfg.get("actions", {}).get(key_specific) or self._cfg.get(
            "actions", {}
        ).get(ev.type)
        if not action_def:
            return
        if not self._cooldown_ok(ev.type):
            return
        # Multi-step action support: steps = [{command, args?} | {task}]
        if "steps" in action_def:
            steps = action_def.get("steps") or []
            # Attach optional context to any task steps
            try:
                if self._context_supplier is not None:
                    ctx = self._context_supplier() or {}
                    if ctx:
                        enriched = []
                        for s in steps:
                            if isinstance(s, dict) and "task" in s:
                                t = str(s["task"]) + f"\n\n[context] {json.dumps(ctx)}"
                                s = {**s, "task": t}
                            enriched.append(s)
                        steps = enriched
            except Exception:
                pass
            if self._loop is None:
                return
            asyncio.run_coroutine_threadsafe(
                self._controller.enqueue(
                    Action(name=ev.type, command="sequence", args={"steps": steps})
                ),
                self._loop,
            )
            return
        # Command-based action (no AI agent)
        if "command" in action_def:
            command = str(action_def["command"]).strip()
            args = action_def.get("args") or {}
            if self._loop is None:
                return
            asyncio.run_coroutine_threadsafe(
                self._controller.enqueue(Action(name=ev.type, command=command, args=args)),
                self._loop,
            )
            return
        # Fallback: Task-based action (AI agent)
        task = str(action_def["task"])
        # Attach optional context (e.g., eye-gaze screen coordinates)
        try:
            if self._context_supplier is not None:
                ctx = self._context_supplier() or {}
                if ctx:
                    task = f"{task}\n\n[context] {json.dumps(ctx)}"
        except Exception:
            # Ignore context errors to avoid dropping actions
            pass
        if self._loop is None:
            # No loop attached; drop the action
            return
        asyncio.run_coroutine_threadsafe(
            self._controller.enqueue(Action(name=ev.type, task=task)),
            self._loop,
        )
