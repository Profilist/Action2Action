from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional
from browser_use import Agent, Browser, ChatOpenAI


@dataclass
class Action:
    name: str
    task: str

class BrowserController:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        executable_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        user_data_dir='%LOCALAPPDATA%\\Google\\Chrome\\User Data'
        profile = "Default"
        self._browser = Browser(executable_path=executable_path, user_data_dir=user_data_dir, profile_directory=profile, keep_alive=True)
        self._model = model
        self._queue: asyncio.Queue[Action] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._running_task: Optional[asyncio.Task[None]] = None

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
                agent = Agent(
                    task=action.task,
                    browser=self._browser,
                    llm=ChatOpenAI(model=self._model),
                )
                await agent.run()
            except Exception as e:  # pragma: no cover
                print(f"[BrowserController] Action '{action.name}' failed: {e}")
                raise e
            finally:
                self._queue.task_done()

    def run_in_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._running_task = loop.create_task(self.start())

    def stop(self) -> None:
        self._stop.set()
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()

    async def enqueue(self, action: Action) -> None:
        await self._queue.put(action)

    async def wait_all(self) -> None:
        await self._queue.join()
