import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import cycle
from typing import Callable, Concatenate, Generator, Iterator, overload

import torch


def consume_futures[T](
    futs: Iterator[Future[T]], n_tasks: int
) -> Generator[T, None, None]:
    pending: deque[Future[T]] = deque()

    for fut in futs:
        pending.append(fut)
        while (pending and pending[0].done()) or len(pending) > n_tasks:
            yield pending.popleft().result()

    for fut in pending:
        yield fut.result()


def iunsqueeze[T](arg_iter: Iterator[T]) -> Iterator[tuple[T]]:
    for arg in arg_iter:
        yield (arg,)


# NOTE: didn't use TypeVarTuple because it isn't contravariant
@overload
def imap[T, U_contra](
    inputs: Iterator[tuple[U_contra]],
    func: Callable[[U_contra], T],
    n_tasks: int,
    *,
    prefetch_factor: int = 2,
) -> Generator[T, None, None]: ...


@overload
def imap[T, U_contra, V_contra](
    inputs: Iterator[tuple[U_contra, V_contra]],
    func: Callable[[U_contra, V_contra], T],
    n_tasks: int,
    *,
    prefetch_factor: int = 2,
) -> Generator[T, None, None]: ...


@overload
def imap[T, U_contra, V_contra, W_contra](
    inputs: Iterator[tuple[U_contra, V_contra, W_contra]],
    func: Callable[[U_contra, V_contra, W_contra], T],
    n_tasks: int,
    *,
    prefetch_factor: int = 2,
) -> Generator[T, None, None]: ...


def imap[T](
    inputs: Iterator[tuple],
    func: Callable[..., T],
    n_tasks: int,  # TODO: rename to n_workers
    *,
    prefetch_factor: int = 2,
) -> Generator[T, None, None]:
    if n_tasks == 0:
        for data_in in inputs:
            yield func(*data_in)
        return
    elif n_tasks < 0:
        n_tasks = os.cpu_count() or 1

    with ThreadPoolExecutor(n_tasks) as executor:
        futs = (executor.submit(func, *data_in) for data_in in inputs)
        yield from consume_futures(futs, n_tasks * prefetch_factor)


# NOTE: didn't use TypeVarTuple because it isn't contravariant
@overload
def imap_multi_gpu[T, U_contra](
    inputs: Iterator[tuple[U_contra]],
    func: Callable[[torch.device, U_contra], T],
    tasks_per_gpu: int = 1,
    *,
    prefetch_factor: int = 2,
) -> Generator[T, None, None]: ...


@overload
def imap_multi_gpu[T, U_contra, V_contra](
    inputs: Iterator[tuple[U_contra, V_contra]],
    func: Callable[[torch.device, U_contra, V_contra], T],
    tasks_per_gpu: int = 1,
    *,
    prefetch_factor: int = 2,
) -> Generator[T, None, None]: ...


@overload
def imap_multi_gpu[T, U_contra, V_contra, W_contra](
    inputs: Iterator[tuple[U_contra, V_contra, W_contra]],
    func: Callable[[torch.device, U_contra, V_contra, W_contra], T],
    tasks_per_gpu: int = 1,
    *,
    prefetch_factor: int = 2,
) -> Generator[T, None, None]: ...


def imap_multi_gpu[T](
    inputs: Iterator[tuple],
    func: Callable[Concatenate[torch.device, ...], T],
    tasks_per_gpu: int = 1,
    *,
    prefetch_factor: int = 2,
) -> Generator[T, None, None]:
    def func_with_gpu(device: torch.device, data_in: tuple) -> T:
        data_out = func(device, *data_in)
        return data_out

    n_gpus = torch.cuda.device_count()
    n_tasks = n_gpus * tasks_per_gpu
    devices = cycle(torch.device(f"cuda:{i}") for i in range(n_gpus))
    yield from imap(
        zip(devices, inputs), func_with_gpu, n_tasks, prefetch_factor=prefetch_factor
    )
