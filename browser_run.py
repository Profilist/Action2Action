import argparse
import asyncio
import threading
import time

from browser_control import BrowserController, Action


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single Browser-Use action (no gestures)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Open https://duckduckgo.com in a new tab and search for 'browser-use'.",
        help="Natural language instruction for the agent.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="LLM model to use (per Browser-Use supported models).",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=30,
        help="Time to wait for the action to complete before shutting down.",
    )
    args = parser.parse_args()

    controller = BrowserController(model=args.model)

    loop = asyncio.new_event_loop()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        controller.run_in_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    # Enqueue a single action
    fut = asyncio.run_coroutine_threadsafe(
        controller.enqueue(Action(name="one_off", task=args.task)),
        loop,
    )
    # Ensure it was enqueued
    fut.result()

    # Give the agent time to run the task
    print(
        f"[browser_run] Action enqueued. Waiting up to {args.wait_seconds}s for completion..."
    )
    time.sleep(args.wait_seconds)

    # Clean shutdown
    controller.stop()
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=5)
    print("[browser_run] Done.")


if __name__ == "__main__":
    main()
