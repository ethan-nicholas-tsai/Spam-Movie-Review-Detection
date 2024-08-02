# timer.py

from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer:
    timers: ClassVar[Dict[str, float]] = {}
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time


import functools


def count_func_time(func):
    """Decorator: Calculate the running time of the program."""
    from datetime import datetime

    @functools.wraps(func)  # Preserve the information of the decorated function.
    def wrapper(*args, **kw):
        start_time = datetime.now()
        res = func(*args, **kw)
        print("[%s] RUN TIME: %s" % (func.__name__, str(datetime.now() - start_time)))
        return res  # Return the return value of the decorated function.

    return wrapper
