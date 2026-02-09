import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import cycle
from typing import Any, Callable, Concatenate, Generator, Iterator, overload

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
