import os
import weakref
from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar, Token
from enum import Enum, auto
from itertools import cycle
from threading import (
    Condition,
    Lock,
    Thread,
    _register_atexit,  # pyrefly: ignore[missing-module-attribute]
)
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, overload

import torch

_finalizers: weakref.WeakKeyDictionary[Any, Callable[[], Any]] = (
    weakref.WeakKeyDictionary()
)
_finalizers_lock = Lock()


if TYPE_CHECKING:
    # since Pyrefly stub will never expose this, type-hint it ourselves
    def _register_atexit[**P](
        func: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
    ) -> None: ...


def _run_finalizers() -> None:
    with _finalizers_lock:
        finalizers = list(_finalizers.values())
    for finalizer in finalizers:
        finalizer()


_register_atexit(_run_finalizers)


_current_exit_stack: ContextVar["PipelineExitStack | None"] = ContextVar(
    "_current_exit_stack", default=None
)


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


class PipelineExitStack(ExitStack[None]):
    def __init__(self) -> None:
        super().__init__()
        self._token: Token[PipelineExitStack | None] | None = None

    def __enter__(self) -> Self:
        if self._token is not None:
            raise RuntimeError("PipelineExitStack is not recursive")
        ret = super().__enter__()
        self._token = _current_exit_stack.set(self)
        return ret

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._token is None:
            # NOTE: if __enter__ sets `self._token` this should be impossible
            raise RuntimeError("missing token for thread-local reset")
        _current_exit_stack.reset(self._token)
        return super().__exit__(exc_type, exc, exc_tb)


class Scheduler[R](Iterator[Future[R]]):
    @classmethod
    def new[*Ts, U](
        cls,
        func: Callable[[*Ts], U],
        args_in: Iterator[tuple[*Ts]],
        n_workers: int,
    ) -> "Scheduler[U]":
        return cls(func, args_in, ThreadPoolExecutor(n_workers))

    def __init__[*Ts](
        self,
        func: Callable[[*Ts], R],
        args_in: Iterator[tuple[*Ts]],
        executor: Executor,  # ownership moves to this instance
    ) -> None:
        futs = (executor.submit(func, *args) for args in args_in)
        weakref.finalize(futs, executor.shutdown, wait=False)
        self._futs = futs
        self._executor = executor
        self._shutdown = False

    def __iter__(self) -> Self:
        _ = self._get_futs_validate()
        return self

    def __next__(self) -> Future[R]:
        return next(self._get_futs_validate())

    def shutdown(self) -> None:
        # NOTE: don't abandon pending futures, args were consumed already
        self._executor.shutdown()
        self._shutdown = True

    def _get_futs_validate(self) -> Iterator[Future[R]]:
        if self._shutdown:
            raise RuntimeError("scheduled with a shut down scheduler")
        return self._futs


class istarmap[R](Iterator[R]):
    def __init__[*Ts](
        self,
        func: Callable[[*Ts], R],
        iterator: Iterator[tuple[*Ts]],
        /,
        *,
        n_workers: int = -1,
    ) -> None:
        if n_workers < 0:
            n_workers = os.cpu_count() or 1
        scheduler = Scheduler.new(func, iterator, n_workers or 1)
        results_iter = consume_futures(scheduler, n_workers)
        self._scheduler = scheduler
        self._results_iter = results_iter
        self._closed = False

        exit_stack = _current_exit_stack.get()
        if exit_stack is not None:
            exit_stack.callback(self.close)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __iter__(self) -> Self:
        _ = self._get_results_iter_validate()
        return self

    def __next__(self) -> R:
        return next(self._get_results_iter_validate())

    def close(self) -> None:
        self._scheduler.shutdown()
        self._closed = True

    def _get_results_iter_validate(self) -> Iterator[R]:
        if self._closed:
            raise RuntimeError("accessed a shut down istarmap")
        return self._results_iter


@overload
def imap[T, U](
    func: Callable[[T], U],
    iterator: Iterator[T],
    /,
    *,
    n_workers: int = ...,
    strict: bool = ...,
) -> istarmap[U]: ...


@overload
def imap[T, U, V](
    func: Callable[[T, U], V],
    iterator: Iterator[T],
    iter2: Iterator[U],
    /,
    *,
    n_workers: int = ...,
    strict: bool = ...,
) -> istarmap[V]: ...


@overload
def imap[T, U, V, W](
    func: Callable[[T, U, V], W],
    iterator: Iterator[T],
    iter2: Iterator[U],
    iter3: Iterator[V],
    /,
    *,
    n_workers: int = ...,
    strict: bool = ...,
) -> istarmap[W]: ...


@overload
def imap[R](
    func: Callable[..., R],
    iterator: Iterator[Any],
    /,
    *iters: Iterator[Any],
    n_workers: int = ...,
    strict: bool = ...,
) -> istarmap[R]: ...


def imap[R](
    func: Callable[..., R],
    iterator: Iterator[Any],  # at least one
    /,
    *iters: Iterator[Any],
    n_workers: int = -1,
    strict: bool = False,
) -> istarmap[R]:
    return istarmap(func, zip(iterator, *iters, strict=strict), n_workers=n_workers)


def imap_multi_gpu[T, U](
    func: Callable[[torch.device, T], U],
    iterator: Iterator[T],
    /,
    *,
    tasks_per_gpu: int = 1,
) -> istarmap[U]:
    # TODO: think about how to extend this project to CPU-only
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise NotImplementedError("CPU-only is currently not handled")
    n_tasks = n_gpus * tasks_per_gpu
    devices = cycle(torch.device(f"cuda:{i}") for i in range(n_gpus))
    return imap(func, devices, iterator, n_workers=n_tasks, strict=False)


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
                lambda: len(self._q) > 0 or self._state is _StreamState.FINISHED
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

        exit_stack = _current_exit_stack.get()
        if exit_stack is not None:
            exit_stack.callback(self.close)

        # to support cancellation when without a context manager, register a
        # finalizer, but shift up on-exit call to _before_ non-daemonic threads
        # are joined (like is done in ThreadPoolExecutor)
        # TODO: open issue on Pyrefly about how finalize stub needs to include
        #       property setter for `atexit` (bypassing the `__slots__` issue),
        #       then remove the error-ignore
        finalizer = weakref.finalize(self, stream.cancel)
        finalizer.atexit = False  # pyrefly: ignore[missing-attribute]
        with _finalizers_lock:
            _finalizers[self] = finalizer

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __iter__(self) -> Self:
        self._start_thread_if_not_started()
        return self

    def __next__(self) -> T:
        self._start_thread_if_not_started()
        item = self._stream.get_result()
        if item is StreamSentinel.token:
            raise StopIteration
        return item

    def close(self) -> None:
        self._stream.cancel(block=self._started)
        if self._started:
            self._thread.join()
        self._finalized = True

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
        except BaseException as e:  # noqa: BLE001  # sends thread's exc downstream
            stream.finish(e)
        else:
            stream.finish(None)
