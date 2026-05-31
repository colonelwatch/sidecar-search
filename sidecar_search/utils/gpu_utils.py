import os
import weakref
from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum, auto
from itertools import cycle
from threading import Condition, Thread
from types import TracebackType
from typing import Any, Concatenate, Self, overload

import torch


def consume_futures[T](
    futs: Iterator[Future[T]], max_pending: int, yield_timeout: float | None = None
) -> Generator[T, None, None]:
    if max_pending < 0:
        raise ValueError("max_pending must be >= 0")

    pending: deque[Future[T]] = deque()

    for fut in futs:
        pending.append(fut)
        while (pending and pending[0].done()) or len(pending) > max_pending:
            yield pending.popleft().result(yield_timeout)

    for fut in pending:
        yield fut.result(yield_timeout)


# NOTE: didn't use TypeVarTuple because it isn't contravariant
@overload
def imap[T, U_contra](
    inputs: Iterator[tuple[U_contra]],
    func: Callable[[U_contra], T],
    n_tasks: int,
    *,
    yield_timeout: float | None = None,
    prefetch_factor: int = 2,
    on_done: Callable[[Future], Any] | None = None,
    on_break: Callable[[Exception | BaseException], Any] | None = None,
) -> Generator[T, None, None]: ...


@overload
def imap[T, U_contra, V_contra](
    inputs: Iterator[tuple[U_contra, V_contra]],
    func: Callable[[U_contra, V_contra], T],
    n_tasks: int,
    *,
    yield_timeout: float | None = None,
    prefetch_factor: int = 2,
    on_done: Callable[[Future], Any] | None = None,
    on_break: Callable[[Exception | BaseException], Any] | None = None,
) -> Generator[T, None, None]: ...


@overload
def imap[T, U_contra, V_contra, W_contra](
    inputs: Iterator[tuple[U_contra, V_contra, W_contra]],
    func: Callable[[U_contra, V_contra, W_contra], T],
    n_tasks: int,
    *,
    yield_timeout: float | None = None,
    prefetch_factor: int = 2,
    on_done: Callable[[Future], Any] | None = None,
    on_break: Callable[[Exception | BaseException], Any] | None = None,
) -> Generator[T, None, None]: ...


# TODO: refactor to a class?
def imap[T](
    inputs: Iterator[tuple],
    func: Callable[..., T],
    n_tasks: int,  # TODO: rename to n_workers
    *,
    yield_timeout: float | None = None,
    prefetch_factor: int = 2,
    on_done: Callable[[Future], Any] | None = None,
    on_break: Callable[[Exception | BaseException], Any] | None = None,
) -> Generator[T, None, None]:
    if prefetch_factor <= 0:
        raise ValueError("invalid prefetch_factor")

    if n_tasks < 0:
        n_tasks = os.cpu_count() or 1

    with ThreadPoolExecutor(n_tasks or 1) as executor:

        def submit[**P](
            func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
        ) -> Future[T]:
            fut = executor.submit(func, *args, **kwargs)
            if on_done:
                fut.add_done_callback(on_done)
            return fut

        try:
            futs = (submit(func, *data_in) for data_in in inputs)
            yield from consume_futures(
                futs, n_tasks * prefetch_factor, yield_timeout=yield_timeout
            )
        except (Exception, BaseException) as e:
            if on_break:
                on_break(e)
            raise


# NOTE: didn't use TypeVarTuple because it isn't contravariant
@overload
def imap_multi_gpu[T, U_contra](
    inputs: Iterator[tuple[U_contra]],
    func: Callable[[torch.device, U_contra], T],
    tasks_per_gpu: int = 1,
    *,
    yield_timeout: float | None = None,
    prefetch_factor: int = 2,
    on_done: Callable[[Future], Any] | None = None,
    on_break: Callable[[Exception | BaseException], Any] | None = None,
) -> Generator[T, None, None]: ...


@overload
def imap_multi_gpu[T, U_contra, V_contra](
    inputs: Iterator[tuple[U_contra, V_contra]],
    func: Callable[[torch.device, U_contra, V_contra], T],
    tasks_per_gpu: int = 1,
    *,
    yield_timeout: float | None = None,
    prefetch_factor: int = 2,
    on_done: Callable[[Future], Any] | None = None,
    on_break: Callable[[Exception | BaseException], Any] | None = None,
) -> Generator[T, None, None]: ...


@overload
def imap_multi_gpu[T, U_contra, V_contra, W_contra](
    inputs: Iterator[tuple[U_contra, V_contra, W_contra]],
    func: Callable[[torch.device, U_contra, V_contra, W_contra], T],
    tasks_per_gpu: int = 1,
    *,
    yield_timeout: float | None = None,
    prefetch_factor: int = 2,
    on_done: Callable[[Future], Any] | None = None,
    on_break: Callable[[Exception | BaseException], Any] | None = None,
) -> Generator[T, None, None]: ...


def imap_multi_gpu[T](
    inputs: Iterator[tuple],
    func: Callable[Concatenate[torch.device, ...], T],
    tasks_per_gpu: int = 1,
    *,
    yield_timeout: float | None = None,
    prefetch_factor: int = 2,
    on_done: Callable[[Future], Any] | None = None,
    on_break: Callable[[Exception | BaseException], Any] | None = None,
) -> Generator[T, None, None]:
    def func_with_gpu(device: torch.device, data_in: tuple) -> T:
        data_out = func(device, *data_in)
        return data_out

    # TODO: think about how to extend this project to CPU-only
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise NotImplementedError("CPU-only is currently not handled")

    n_tasks = n_gpus * tasks_per_gpu
    devices = cycle(torch.device(f"cuda:{i}") for i in range(n_gpus))
    yield from imap(
        zip(devices, inputs),
        func_with_gpu,
        n_tasks,
        yield_timeout=yield_timeout,
        prefetch_factor=prefetch_factor,
        on_done=on_done,
        on_break=on_break,
    )


class _StreamState(Enum):
    RUNNING = auto()
    CANCELLING = auto()
    FINISHED = auto()


# TODO: use typing_extensions.Sentinel?
class StreamSentinel(Enum):
    token = auto()


class StreamSlot[T]:
    def __init__(self, q: deque[T], cv: Condition) -> None:
        self._q = q
        self._cv = cv
        self._closed = False

    def put_result(self, result: T) -> None:
        if self._closed:
            raise RuntimeError("accessed slot after it was closed")
        with self._cv:
            self._q.append(result)
            self._cv.notify_all()
        self.close()

    def close(self) -> None:
        self._closed = True


class Stream[T]:
    @classmethod
    def new(cls, maxsize: int = 0) -> Self:
        q: deque[T] = deque()
        cv: Condition = Condition()
        return cls(maxsize, q, cv)

    def __init__(self, maxsize: int, q: deque[T], cv: Condition) -> None:
        self._maxsize = maxsize
        self._q = q
        self._state = _StreamState.RUNNING
        self._exc: BaseException | None = None
        self._cv = cv

    @contextmanager
    def wait_for_slot_or_cancelling(
        self,
    ) -> Generator[StreamSlot[T] | None, None, None]:
        with self._cv:
            if self._state is _StreamState.FINISHED:
                raise RuntimeError("waited for slot in a finished stream")
            if self._maxsize > 0:
                self._cv.wait_for(
                    lambda: (
                        len(self._q) < self._maxsize
                        or self._state is _StreamState.CANCELLING
                    )
                )
            cancelling = self._state is _StreamState.CANCELLING

        if cancelling:
            yield None
            return

        slot = StreamSlot(self._q, self._cv)
        try:
            yield slot
        finally:
            slot.close()

    def finish(self, exc: BaseException | None) -> None:
        with self._cv:
            if self._state is _StreamState.FINISHED:
                raise RuntimeError("finished a stream twice")
            self._exc = exc
            self._state = _StreamState.FINISHED
            self._cv.notify_all()

    def get_result(self) -> T | StreamSentinel:
        with self._cv:
            self._cv.wait_for(
                lambda: (len(self._q) > 0 or self._state is _StreamState.FINISHED)
            )
            if len(self._q) > 0:
                result = self._q.popleft()
                self._cv.notify_all()
            elif self._exc is None:
                result = StreamSentinel.token
            else:
                raise self._exc
        return result

    def cancel(self, block: bool = False) -> None:
        with self._cv:
            if self._state is _StreamState.FINISHED:
                return
            self._state = _StreamState.CANCELLING
            self._cv.notify_all()
            if block:
                self._cv.wait_for(lambda: self._state is _StreamState.FINISHED)


class iqueue[T](Iterator[T]):
    def __init__(self, items: Iterator[T], maxsize: int = 4) -> None:
        stream: Stream[T] = Stream.new(maxsize)
        self._stream = stream
        self._started = False
        self._finalized = False
        self._thread = Thread(target=self._routine, args=(stream, items))
        weakref.finalize(self, stream.cancel)

    def __enter__(self) -> Self:
        self._start_thread_if_not_started()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType,
    ) -> None:
        # cancel stream manually, and to have a gc-like lifetime, bar future use
        # of this object by setting (and checking) self._finalized
        assert self._started, "__enter__ did not start thread"
        self._stream.cancel(block=True)
        self._thread.join()
        self._finalized = True

    def __iter__(self) -> Self:
        self._start_thread_if_not_started()
        return self

    def __next__(self) -> T:
        self._start_thread_if_not_started()
        item = self._stream.get_result()
        if item is StreamSentinel.token:
            raise StopIteration
        return item

    def _start_thread_if_not_started(self) -> None:
        if self._finalized:
            raise RuntimeError("used an iqueue that was shut down")
        if not self._started:
            self._thread.start()
            self._started = True

    @staticmethod
    def _routine(stream: Stream[T], items: Iterator[T]) -> None:
        try:
            while True:
                with stream.wait_for_slot_or_cancelling() as slot:
                    if not slot:
                        break  # we are cancelling
                    try:
                        item = next(items)
                    except StopIteration:
                        break
                    else:
                        slot.put_result(item)
        except BaseException as e:
            stream.finish(e)
        else:
            stream.finish(None)
