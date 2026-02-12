from concurrent.futures import Future
from itertools import count, cycle
from threading import Event
from typing import Any, Callable, Generator, Iterator, Never, TypedDict, cast
from unittest.mock import ANY, MagicMock, create_autospec

import pytest
import torch

from sidecar_search.utils.gpu_utils import consume_futures, imap, imap_multi_gpu


class TestConsumeFutures:
    # NOTE: max_pending = 1 tests exhausting futs iterator before blocking
    @pytest.mark.parametrize("max_pending", [0, 1])
    def test_blocking(self, max_pending: int) -> None:
        fut: Future[Never] = Future()
        futs = iter((fut,))
        with pytest.raises(TimeoutError):
            _ = list(consume_futures(futs, max_pending, yield_timeout=0))

    def test_backpressure(self) -> None:
        fut_0: Future[Never] = Future()
        fut_1: Future[Never] = Future()
        futs = iter((fut_0, fut_1))

        try:
            _ = list(consume_futures(futs, 0, yield_timeout=0))
        except TimeoutError:
            pass

        assert next(futs) is fut_1

    def test_yield_order(self) -> None:
        fut_0: Future[int] = Future()
        fut_1: Future[int] = Future()

        def reverse_resolution() -> Generator[Future[int], None, None]:
            fut_1.set_result(1)
            yield fut_0
            fut_0.set_result(0)
            yield fut_1

        assert list(consume_futures(reverse_resolution(), 2)) == [0, 1]

    def test_exception_propagation(self) -> None:
        exc = Exception()
        fut: Future[Never] = Future()
        fut.set_exception(exc)

        futs = iter((fut,))
        with pytest.raises(Exception) as raised:
            _ = list(consume_futures(futs, 0))

        assert raised.value is exc

    def test_raise_on_negative_max_pending(self) -> None:
        def noop_futures() -> Generator[Future[None], None, None]:
            for _ in range(10):
                fut = Future()
                fut.set_result(None)
                yield fut

        with pytest.raises(ValueError):
            _ = list(consume_futures(noop_futures(), -1))

    def test_empty_iterator(self) -> None:
        futs: Iterator[Future] = iter(tuple())
        assert list(consume_futures(futs, 0)) == []


class TestImapParallel:
    def test_map(self) -> None:
        def double(x: int) -> int:
            return x * 2

        vals = range(10)
        results_iter = imap(zip(vals), double, 2)
        assert list(map(double, vals)) == list(results_iter)

    def test_order(self) -> None:
        event = Event()

        def on_done(fut: Future) -> None:
            if fut.result() == 1:
                event.set()

        def func(i: int) -> int:
            if i == 0:
                event.wait()
            return i

        vals = range(2)
        assert list(imap(zip(vals), func, 2, on_done=on_done)) == [0, 1]


@pytest.mark.parametrize("n_tasks", [0, 1])
class TestImapConcurrent:
    def test_blocking(self, n_tasks: int) -> None:
        evt = Event()

        with pytest.raises(TimeoutError):
            vals = range(10)
            results_iter = imap(
                zip(vals),
                lambda _: evt.wait(),
                n_tasks,
                yield_timeout=0,
                on_break=(lambda _: evt.set()),
            )
            _ = list(results_iter)

    @pytest.mark.parametrize("prefetch_factor", [1, 2])
    def test_backpressure(self, n_tasks: int, prefetch_factor: int) -> None:
        evt = Event()

        vals_iter = count()
        try:
            results_iter = imap(
                zip(vals_iter),
                lambda _: evt.wait(),
                n_tasks,
                yield_timeout=0,
                prefetch_factor=prefetch_factor,
                on_break=(lambda _: evt.set()),
            )
            _ = list(results_iter)
        except TimeoutError:
            pass

        assert next(vals_iter) == n_tasks * prefetch_factor + 1

    def test_on_break(self, n_tasks: int) -> None:
        exc = Exception()
        passed: Exception | BaseException | None = None
        raised: Exception | BaseException | None = None

        def raise_exc(_) -> None:
            raise exc

        def on_break(e: Exception | BaseException) -> None:
            nonlocal passed
            passed = e

        vals = range(10)
        try:
            _ = list(imap(zip(vals), raise_exc, n_tasks, on_break=on_break))
        except Exception as e:
            raised = e

        assert exc is passed and exc is raised

    def test_raise_on_nonpositive_prefetch_factor(self, n_tasks: int) -> None:
        with pytest.raises(ValueError):
            vals = range(10)
            _ = list(imap(zip(vals), lambda x: x, n_tasks, prefetch_factor=-1))

    def test_empty_iterator(self, n_tasks: int) -> None:
        vals = iter(tuple())
        assert list(imap(zip(vals), lambda x: x, n_tasks)) == []


@pytest.fixture
def mock_imap(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_imap = cast(MagicMock, create_autospec(spec=imap))  # duck-typing as MagicMock
    mock_imap.side_effect = lambda *args, **kwargs: (x for x in iter([]))
    monkeypatch.setattr("sidecar_search.utils.gpu_utils.imap", mock_imap)
    return mock_imap


# NOTE: these are kw-only arguments, which don't require manual resolution
class ExpectedImapKwargs(TypedDict):
    yield_timeout: float | None
    prefetch_factor: int
    on_done: Callable[[Future], Any] | None
    on_break: Callable[[Exception | BaseException], Any] | None


@pytest.mark.usefixtures("mock_gpu_env")
class TestImapMultiGpu:
    def test_args_concatenate(self) -> None:
        vals = range(10)
        idxs = cycle(range(torch.cuda.device_count()))

        def combine(device: torch.device, value: int) -> tuple[int, int]:
            return (device.index, value)

        results = list(imap_multi_gpu(zip(vals), combine))
        expecteds = list(zip(idxs, vals))
        assert results == expecteds

    def test_tasks_arg_passed(self, mock_imap: MagicMock) -> None:
        vals = range(10)
        n_tasks_per_gpu = 2
        _ = list(imap_multi_gpu(zip(vals), (lambda _, x: x), n_tasks_per_gpu))

        mock_imap.assert_called_once()

        try:
            n_tasks = mock_imap.call_args.kwargs["n_tasks"]
        except KeyError:
            n_tasks = mock_imap.call_args.args[2]

        assert n_tasks_per_gpu * torch.cuda.device_count() == n_tasks

    def test_kwargs_passed(self, mock_imap: MagicMock) -> None:
        vals = range(10)
        my_kwargs = ExpectedImapKwargs(
            yield_timeout=10,
            prefetch_factor=3,
            on_done=(lambda _: None),
            on_break=(lambda _: None),
        )
        _ = list(imap_multi_gpu(zip(vals), lambda _, x: x, **my_kwargs))
        mock_imap.assert_called_once_with(ANY, ANY, ANY, **my_kwargs)
