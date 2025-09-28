from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any
from browser_use import Agent, Browser, ChatOpenAI
import pyautogui
import json
import re
try:
    import pygetwindow as gw  # type: ignore
except Exception:  # pragma: no cover
    gw = None  # Fallback if pygetwindow is unavailable


@dataclass
class Action:
    name: str
    task: Optional[str] = None
    command: Optional[str] = None
    args: Optional[Dict[str, Any]] = None

class BrowserController:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        executable_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        user_data_dir='%LOCALAPPDATA%\\Google\\Chrome\\User Data'
        profile = "Default"
        self._browser = Browser(executable_path=executable_path, user_data_dir=user_data_dir, profile_directory=profile, keep_alive=True)
        self._model = model
        # Single-action policy: do not build up a backlog
        # We still keep a queue structure for simplicity, but we only accept
        # an action when the controller is idle.
        self._queue: asyncio.Queue[Action] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._running_task: Optional[asyncio.Task[None]] = None
        self._busy: bool = False

    async def start(self) -> None:
        # Ensure a single persistent browser session for all actions
        try:
            await self._browser.start()
        except Exception:
            # If already started, continue
            pass
        while not self._stop.is_set():
            action = await self._queue.get()
            try:
                if self._browser is None:
                    print(
                        f"[BrowserController] Dependencies not available. Skipping action: {action.name}"
                    )
                    continue
                # If a direct command is provided, execute without AI agent
                if action.command:
                    self._busy = True
                    await self._execute_command(action.command, action.args or {})
                else:
                    agent = Agent(
                        task=action.task or "",
                        browser=self._browser,
                        llm=ChatOpenAI(model=self._model),
                    )
                    # Mark busy to block new enqueues while this action runs
                    self._busy = True
                    await agent.run()
            except Exception as e:  # pragma: no cover
                print(f"[BrowserController] Action '{action.name}' failed: {e}")
                raise e
            finally:
                self._queue.task_done()
                self._busy = False

    def run_in_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._running_task = loop.create_task(self.start())

    def stop(self) -> None:
        self._stop.set()
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()

    async def enqueue(self, action: Action) -> bool:
        """Attempt to enqueue an action.

        Returns True if accepted. Returns False if the controller is busy or
        an action is already queued (single-action policy to avoid backlogs).
        """
        # If currently executing or there is already a pending action, drop.
        if self._busy or not self._queue.empty():
            return False
        await self._queue.put(action)
        return True

    async def wait_all(self) -> None:
        await self._queue.join()

    async def _execute_command(self, command: str, args: Dict[str, Any]) -> None:
        """Execute a direct browser command without invoking the AI agent."""
        try:
            if command == "sequence":
                steps = args.get("steps") or []
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    if "command" in step:
                        await self._execute_command(str(step["command"]), step.get("args") or {})
                    elif "task" in step:
                        agent = Agent(
                            task=str(step["task"]),
                            browser=self._browser,
                            llm=ChatOpenAI(model=self._model),
                        )
                        await agent.run()
                return
            if command == "next_slide":
                self._focus_browser_window()
                pyautogui.press("down")
            if command == "previous_slide":
                self._focus_browser_window()
                pyautogui.press("up")
            if command == "open_slides":
                self._focus_browser_window()
                pyautogui.hotkey("ctrl", "l")
                pyautogui.typewrite(args.get("url", ""))
                pyautogui.press("enter")
                await asyncio.sleep(float(args.get("wait_open_s", 5.0)))
                pyautogui.hotkey("ctrl", "f5")
            if command in ("tab_next", "tab_prev"):
                self._focus_browser_window()
                if command == "tab_next":
                    pyautogui.hotkey("ctrl", "tab")
                else:
                    pyautogui.hotkey("ctrl", "shift", "tab")
            elif command == "tab_new":
                self._focus_browser_window()
                pyautogui.hotkey("ctrl", "l")
                pyautogui.typewrite(args.get("url", ""))
                pyautogui.press("enter")
            elif command == "tab_open":
                self._focus_browser_window()
                pyautogui.hotkey("ctrl", "t")
                pyautogui.typewrite(args.get("url", ""))
                pyautogui.press("enter")
            
            elif command == "tab_index":
                index = int(args.get("index", 1))
                if 1 <= index <= 8:
                    self._focus_browser_window()
                    # Ctrl+<number> to jump to specific tab
                    pyautogui.hotkey("ctrl", str(index))
            elif command == "tab_close":
                self._focus_browser_window()
                pyautogui.hotkey("ctrl", "w")
            elif command == "open_patient":
                self._focus_browser_window()
                pyautogui.hotkey("ctrl", "t")
                pyautogui.typewrite(args.get("url", ""))
                pyautogui.press("enter")
            elif command == "event_to_calendar":
                # 1) Extract event details from current page
                extract_task = (
                    "Scan the PAGE https://luma.com/techto-studenthack-sep-27-2025 and extract event details: "
                    "title, start_date, start_time, end_date, end_time, location. "
                    "Return ONLY a minified JSON object with exactly these keys and no extra text."
                )
                agent = Agent(
                    task=extract_task,
                    browser=self._browser,
                    llm=ChatOpenAI(model=self._model),
                )
                history = await agent.run(max_steps=12)
                result_text = None
                try:
                    # Prefer documented API
                    if hasattr(history, "final_result"):
                        result_text = history.final_result()  # type: ignore[attr-defined]
                except Exception:
                    result_text = None
                if not result_text and isinstance(history, str):
                    result_text = history
                details = {
                    "title": "",
                    "start_date": "",
                    "start_time": "",
                    "end_date": "",
                    "end_time": "",
                    "location": "",
                }
                if isinstance(result_text, str) and result_text.strip():
                    try:
                        details = {**details, **json.loads(result_text)}
                    except Exception:
                        m = re.search(r"\{[\s\S]*\}", result_text)
                        if m:
                            try:
                                details = {**details, **json.loads(m.group(0))}
                            except Exception:
                                pass
                # Back-compat: if only 'date'/'time' present
                if not details.get("start_date") and details.get("date"):
                    details["start_date"] = str(details.get("date", ""))
                if not details.get("start_time") and details.get("time"):
                    details["start_time"] = str(details.get("time", ""))
                if not details.get("end_date") and details.get("start_date"):
                    details["end_date"] = str(details.get("start_date"))

                title = str(details.get("title", "")).strip()
                start_date = str(details.get("start_date", "")).strip()
                start_time = str(details.get("start_time", "")).strip()
                end_time = str(details.get("end_time", "")).strip()
                end_date = str(details.get("end_date", "")).strip()
                location = str(details.get("location", "")).strip()

                # 2) Create the event in Google Calendar using pyautogui per specified tab order
                self._focus_browser_window()
                pyautogui.hotkey("ctrl", "l")
                pyautogui.typewrite("https://calendar.google.com/calendar/u/0/r/eventedit")
                pyautogui.press("enter")
                await asyncio.sleep(float(args.get("wait_open_s", 2.0)))

                # Title field focused
                if title:
                    pyautogui.typewrite(title)
                await asyncio.sleep(0.15)

                # One Tab for Save, next is Date
                pyautogui.press("tab")  # Save (skip)
                await asyncio.sleep(0.05)
                pyautogui.press("tab")  # Start date
                await asyncio.sleep(0.05)
                if start_date:
                    pyautogui.typewrite(start_date)
                await asyncio.sleep(0.1)

                # Next -> Start time
                pyautogui.press("tab")
                await asyncio.sleep(0.05)
                if start_time:
                    pyautogui.typewrite(start_time)
                await asyncio.sleep(0.1)

                # Next -> End time
                pyautogui.press("tab")
                await asyncio.sleep(0.05)
                if end_time:
                    pyautogui.typewrite(end_time)
                await asyncio.sleep(0.1)

                # Next -> End date
                pyautogui.press("tab")
                await asyncio.sleep(0.05)
                if end_date:
                    pyautogui.typewrite(end_date)
                await asyncio.sleep(0.1)

                # 11 tabs to Location
                for _ in range(int(args.get("tabs_to_location", 11))):
                    pyautogui.press("tab")
                    await asyncio.sleep(0.03)
                if location:
                    pyautogui.typewrite(location)
                await asyncio.sleep(0.1)

                # 27 tabs to Save
                for _ in range(int(args.get("tabs_to_save", 29))):
                    pyautogui.press("tab")
                    await asyncio.sleep(0.03)
                pyautogui.press("enter")
            elif command == "nothing":
                pass
            else:
                print(f"[BrowserController] Unknown command: {command}")
        except Exception as e:  # pragma: no cover
            print(f"[BrowserController] Command '{command}' failed: {e}")

    def _focus_browser_window(self) -> None:
        """Best-effort attempt to focus the Chrome window so hotkeys apply there."""
        try:
            if gw is None:
                return
            # Try common Chrome window titles
            candidates = [
                "Google Chrome",
                "Chrome",
            ]
            windows = []
            for title in candidates:
                windows.extend(gw.getWindowsWithTitle(title))
            if not windows:
                return
            # Prefer a non-minimized window
            win = None
            for w in windows:
                try:
                    if not w.isMinimized:
                        win = w
                        break
                except Exception:
                    continue
            win = win or windows[0]
            try:
                win.activate()
            except Exception:
                pass
        except Exception:
            # Ignore focusing errors silently
            pass
