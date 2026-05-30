from collections import deque
from itertools import count
from threading import Condition, Event, Lock, RLock
from time import monotonic
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable

import pytest

type Predicate[T] = Callable[[], T]


class CapturingCondition(Condition):
    def __init__(self, lock: "Lock | RLock | None" = None) -> None:
        if lock is not None:  # preserve signature out of LSP, but also fail
            # the problem is that hooks can't be run on the underlying lock
            pytest.fail("separate access to owned lock is not supported")

        super().__init__(lock)

        # Underlying Condition class defines `acquire` and `release` via
        # attributes, so override upon inheritance wouldn't work.
        self._acquire_impl = self.acquire
        self._release_impl = self.release
        self.acquire = self._acquire
        self.release = self._release

        # covered by condition variable lock
        self._predicate_states: dict[int, tuple[Predicate[Any], Event]] = {}
        self._notified: bool = False

        # covered by _capture_lock to ensure atomic capture operations
        self._counter = count()  # incidental; iterators aren't thread-safe
        self._on_block: dict[Predicate[Any], Callable[[], Any]] = {}
        self._capture_lock = Lock()

    def call_on_next_block(
        self,
        on_block: Callable[[], Any],
        trigger_predicate: Predicate[Any] = lambda: True,
        overwrite: bool = False,
    ) -> None:
        with self._capture_lock:
            if trigger_predicate in self._on_block and not overwrite:
                pytest.fail("queued same trigger_predicate before it could be used")
            self._on_block[trigger_predicate] = on_block

    def wait(self, timeout: float | None = None, *, private: bool = False) -> bool:
        # maintaining state through a loop gets unwieldy
        if not private:
            pytest.fail("traditional wait syntax is not supported, use wait_for")
        return super().wait(timeout)

    def wait_for[T](
        self,
        predicate: Predicate[T],
        timeout: float | None = None,
        *,
        private: bool = False,
    ) -> T:
        if not self._is_owned():  # move up this check to prevent deadlock
            raise RuntimeError("cannot wait on an un-acquired lock")

        if private:
            return self._wait_for_private(predicate, timeout=timeout)

        with self._capture_lock:
            active_trigger_predicate = None
            for trigger_predicate, on_block in self._on_block.items():
                if not trigger_predicate():
                    continue
                if active_trigger_predicate is None:
                    active_trigger_predicate = trigger_predicate
                else:
                    pytest.fail("more than one trigger predicate fired")

            if active_trigger_predicate is not None:
                state = (
                    next(self._counter),
                    self._on_block.pop(active_trigger_predicate),
                    Event(),
                )
            else:
                state = None

        if state is None:
            return self._wait_for_private(predicate, timeout=timeout)

        # check if the predicate is already True
        value = predicate()
        if value:
            return value

        handle, on_block, could_advance = state
        self._predicate_states[handle] = (predicate, could_advance)
        try:
            self.release(private=True)
            try:
                on_block()
            finally:
                self.acquire(private=True)
        finally:
            del self._predicate_states[handle]

        # check predicate one last time in case it flipped back to False
        value = predicate()
        if not (value and could_advance.is_set()):
            pytest.fail("could block forever")
        return value

    # NOTE: taken from underlying Condition source code
    def _wait_for_private[T](
        self,
        predicate: Predicate[T],
        timeout: float | None = None,
    ) -> T:
        endtime = None
        waittime = timeout
        result = predicate()
        while not result:
            if waittime is not None:
                if endtime is None:
                    endtime = monotonic() + waittime
                else:
                    waittime = endtime - monotonic()
                    if waittime <= 0:
                        break
            self.wait(waittime, private=True)
            result = predicate()
        return result

    def _acquire(  # NOTE: replaces `self.acquire`
        self, blocking: bool = True, timeout: float = -1, *, private: bool = False
    ) -> bool:
        if not private:
            pytest.fail("direct acquire is not supported, use context manager")
        return self._acquire_impl(blocking=blocking, timeout=timeout)

    def _release(  # NOTE: replaces `self.release`
        self, *, private: bool = False
    ) -> None:
        if not private:
            pytest.fail("direct release is not supported, use context manager")
        self._release_impl()

    def notify(self, n: int = 1, *, private: bool = False) -> None:
        # working with the threads that super().notify woke up is unwieldy, and
        # using notify_all instead should be valid in a correct program
        if not private:
            pytest.fail("waking individual waiters is not supported, use notify_all")
        super().notify(n=n)

    def notify_all(self) -> None:
        if not self._is_owned():
            # notify_all also checks, but it's explicitly needed at this level
            # to guarantee atomicity
            raise RuntimeError("cannot notify on un-acquired lock")
        self.notify(n=len(self._waiters), private=True)
        self._notified = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # normal condition variable use notifies and makes the predicate true
        # atomically, so this checks that
        try:
            if self._notified:
                for predicate, could_advance in self._predicate_states.values():
                    if predicate():
                        could_advance.set()
                self._notified = False
        finally:
            super().__exit__(exc_type, exc_val, exc_tb)

    if TYPE_CHECKING:
        _waiters: deque["Lock"]

        def _is_owned(self) -> bool: ...
        def acquire(
            self, blocking: bool = True, timeout: float = -1, *, private: bool = False
        ) -> bool: ...
        def release(self, *, private: bool = False) -> None: ...
