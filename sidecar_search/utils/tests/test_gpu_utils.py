import gc
import os
import re
from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from itertools import cycle
from threading import Condition, Event
from typing import Any, Never, Protocol, cast
from unittest.mock import (
    MagicMock,
    NonCallableMagicMock,
    call,
    create_autospec,
    sentinel,
)

import pytest
import torch

from .. import gpu_utils
from ..gpu_utils import (
    PipelineExitStack,
    Scheduler,
    Stream,
    StreamSentinel,
    StreamSlot,
    consume_futures,
    imap,
    imap_multi_gpu,
    iqueue,
    istarmap,
)
from .capturing_condition import CapturingCondition

_N_CPUS = os.cpu_count() or 1


class _SupportsClose(Protocol):
    def close(self) -> None: ...


def _step_close(resource: _SupportsClose) -> None:
    resource.close()


def _step_context_manager(ctx: AbstractContextManager[Any]) -> None:
    with ctx:
        pass


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
        futs: Iterator[Never] = iter(())
        assert list(consume_futures(futs, 0)) == []


class TestScheduler:
    def test_coupled_consumption(self) -> None:
        vals_iter = iter(range(10))
        scheduler = Scheduler.new(lambda x: x, zip(vals_iter), 2)
        _ = next(scheduler)
        actual = next(vals_iter)
        assert actual == 1  # pre-consumption poses in-flight risk

    def test_submit(self) -> None:
        def identity(x: Any) -> Any:
            return x

        vals = range(10)
        mock_executor: NonCallableMagicMock = create_autospec(
            ThreadPoolExecutor, instance=True
        )
        scheduler = Scheduler(identity, zip(vals), mock_executor)
        _ = list(scheduler)

        assert mock_executor.submit.call_args_list == [call(identity, x) for x in vals]

    def test_empty_iterator(self) -> None:
        vals: Iterator[Never] = iter(())
        scheduler = Scheduler.new(lambda x: x, zip(vals), 1)
        actual = list(consume_futures(scheduler, 1))
        assert actual == []

    def test_shutdown_delegates_to_executor(self) -> None:
        mock_executor: NonCallableMagicMock = create_autospec(
            ThreadPoolExecutor, instance=True
        )
        scheduler = Scheduler(lambda x: x, zip(range(10)), mock_executor)
        scheduler.shutdown()
        mock_executor.shutdown.assert_called_once_with()

    @pytest.mark.parametrize("action", [iter, next])
    def test_raises_after_shutdown(
        self, action: Callable[[Scheduler[Any]], Any]
    ) -> None:
        scheduler = Scheduler.new(lambda x: x, zip(range(10)), 1)
        scheduler.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            _ = action(scheduler)

    def test_shutdown_on_gc(self) -> None:
        mock_executor: NonCallableMagicMock = create_autospec(
            ThreadPoolExecutor, instance=True
        )
        scheduler = Scheduler(lambda x: x, zip(range(10)), mock_executor)
        del scheduler
        gc.collect()
        mock_executor.shutdown.assert_called_once_with(wait=False)


class TestIstarmap:
    @pytest.mark.parametrize(
        ("n_workers", "expected_n_workers", "expected_n_pending"),
        [
            pytest.param(-1, _N_CPUS, _N_CPUS, id="all_cores"),
            pytest.param(0, 1, 0, id="synchronous"),
            pytest.param(1, 1, 1, id="deferred"),
            pytest.param(2, 2, 2, id="multicore"),
        ],
    )
    def test_istarmap(
        self,
        n_workers: int,
        expected_n_workers: int,
        expected_n_pending: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_scheduler_new: MagicMock = create_autospec(Scheduler.new)
        patch_scheduler_new.return_value = sentinel.scheduler
        monkeypatch.setattr(Scheduler, "new", patch_scheduler_new)

        patch_consume_futures: MagicMock = create_autospec(consume_futures)
        patch_consume_futures.return_value = iter(range(10))
        monkeypatch.setattr(gpu_utils, "consume_futures", patch_consume_futures)

        def identity(x: Any) -> Any:
            return x

        actual = list(istarmap(identity, sentinel.args_in, n_workers=n_workers))

        # transitively prove properties by proving forwarding
        patch_scheduler_new.assert_called_once_with(
            identity, sentinel.args_in, expected_n_workers
        )
        patch_consume_futures.assert_called_once_with(
            sentinel.scheduler, expected_n_pending
        )

        assert actual == list(range(10))

    @pytest.mark.parametrize(
        ("close_action", "action"),
        [
            (_step_close, iter),
            (_step_close, next),
            (_step_context_manager, next),
        ],
    )
    def test_raises_on_access_after_close(
        self,
        close_action: Callable[[istarmap[Any]], Any],
        action: Callable[[istarmap[Any]], Any],
    ) -> None:
        items_recv_iter = istarmap(lambda x: x, zip(range(10)), n_workers=4)
        close_action(items_recv_iter)
        with pytest.raises(RuntimeError, match="shut down"):
            _ = action(items_recv_iter)

    def test_exit_stack(self) -> None:
        exit_stack = PipelineExitStack()
        with exit_stack:
            im = istarmap(lambda x: x, zip(range(10)), n_workers=1)
        with pytest.raises(RuntimeError, match="shut down"):
            _ = next(im)


@pytest.mark.integration
@pytest.mark.parametrize("n_workers", [-1, 0, 1, 2])
def test_istarmap_integration(n_workers: int) -> None:
    actual = list(istarmap(lambda x: 2 * x, zip(range(10)), n_workers=n_workers))
    assert actual == list(range(0, 20, 2))


@pytest.fixture
def mock_istarmap(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_istarmap = cast(
        MagicMock, create_autospec(spec=istarmap)
    )  # duck-typing as MagicMock
    mock_istarmap.side_effect = lambda *args, **kwargs: (x for x in iter([]))
    monkeypatch.setattr(gpu_utils, "istarmap", mock_istarmap)
    return mock_istarmap


class TestImap:
    @pytest.mark.parametrize(
        ("primes", "func", "expected"),
        [
            (((2, 3),), lambda x: x, [2, 3]),
            (((2, 3), (5, 7)), lambda x, y: x * y, [10, 21]),
            (((2, 3), (5, 7), (11, 13)), lambda x, y, z: x * y * z, [110, 273]),
        ],
    )
    def test_zip(
        self,
        primes: tuple[tuple[int, ...], ...],
        func: Callable[[*tuple[int, ...]], int],
        expected: list[int],
    ) -> None:
        actual = list(imap(func, *(iter(x) for x in primes), strict=True))
        assert actual == expected

    def test_not_strict(self) -> None:
        vals_iter_0 = iter((2, 3, 5))
        vals_iter_1 = iter((7, 11))
        actual = list(imap(lambda x, y: x * y, vals_iter_0, vals_iter_1, strict=False))
        assert actual == [14, 33]

    def test_strict(self) -> None:
        vals_iter_0 = iter((2, 3, 5))
        vals_iter_1 = iter((7, 11))
        with pytest.raises(
            ValueError,
            match=re.escape("zip() argument 2 is shorter than argument 1"),
        ):
            _ = list(imap(lambda x, y: x * y, vals_iter_0, vals_iter_1, strict=True))

    def test_n_workers_arg_passed(self, mock_istarmap: MagicMock) -> None:
        vals = range(10)
        n_workers = 7
        _ = list(imap((lambda x: x), iter(vals), n_workers=n_workers))
        mock_istarmap.assert_called_once()
        actual_n_workers = mock_istarmap.call_args.kwargs.get("n_workers", None)
        assert n_workers == actual_n_workers


@pytest.mark.usefixtures("mock_gpu_env")
class TestImapMultiGpu:
    def test_args_concatenate(self) -> None:
        vals = range(10)
        idxs = cycle(range(torch.cuda.device_count()))

        def combine(device: torch.device, value: int) -> tuple[int, int]:
            return (device.index, value)

        results = list(imap_multi_gpu(combine, iter(vals)))
        expecteds = list(zip(idxs, vals))
        assert results == expecteds

    def test_tasks_arg_passed(self, mock_istarmap: MagicMock) -> None:
        vals = range(10)
        n_tasks_per_gpu = 2
        _ = list(
            imap_multi_gpu((lambda _, x: x), iter(vals), tasks_per_gpu=n_tasks_per_gpu)
        )
        mock_istarmap.assert_called_once()
        n_tasks = mock_istarmap.call_args.kwargs["n_workers"]
        assert n_tasks_per_gpu * torch.cuda.device_count() == n_tasks


@pytest.fixture
def mock_q() -> NonCallableMagicMock:
    return create_autospec(deque, instance=True)


@pytest.fixture
def mock_cv() -> NonCallableMagicMock:
    return create_autospec(Condition, instance=True)


@pytest.fixture
def parent_mock(
    mock_q: NonCallableMagicMock, mock_cv: NonCallableMagicMock
) -> MagicMock:
    parent = MagicMock()
    parent.attach_mock(mock_q, "q")
    parent.attach_mock(mock_cv, "cv")
    return parent


class TestStreamSlot:
    def test_put_result(self, parent_mock: MagicMock) -> None:
        slot = StreamSlot(parent_mock.q, parent_mock.cv)
        slot.put_result(42)
        parent_mock.assert_has_calls(
            [
                call.cv.__enter__(),
                call.q.append(42),
                call.cv.notify_all(),
                call.cv.__exit__(None, None, None),
            ]
        )

    def test_put_result_raises_on_second_call(
        self, mock_cv: NonCallableMagicMock
    ) -> None:
        slot: StreamSlot[int] = StreamSlot(deque(), mock_cv)
        slot.put_result(1)
        with pytest.raises(RuntimeError, match="closed"):
            slot.put_result(2)

    def test_put_result_raises_on_call_after_close(
        self, mock_cv: NonCallableMagicMock
    ) -> None:
        slot: StreamSlot[int] = StreamSlot(deque(), mock_cv)
        slot.close()
        with pytest.raises(RuntimeError, match="closed"):
            slot.put_result(42)


@pytest.mark.integration
def test_stream_slot_integration_put_result() -> None:
    q: deque[int] = deque([1])
    slot = StreamSlot(q, Condition())
    slot.put_result(2)
    assert list(q) == [1, 2]


class SentinelException(BaseException):
    pass


class TestStream:
    def test_wait_for_slot(self, mock_cv: NonCallableMagicMock) -> None:
        q: deque[int] = deque()
        stream = Stream(0, q, mock_cv)

        with stream.wait_for_slot_or_cancelling() as slot:
            if not slot:
                pass  # unexpected: stream is cancelling
            else:
                slot.put_result(42)

        assert list(q) == [42]

    def test_wait_for_slot_would_block(self) -> None:
        q: deque[int] = deque([1])
        cv = CapturingCondition()
        stream = Stream(len(q), q, cv)

        blocked = Event()

        def on_block() -> None:
            blocked.set()
            _ = stream.get_result()

        cv.call_on_next_block(on_block)

        with stream.wait_for_slot_or_cancelling() as slot:
            pass

        assert slot is not None
        assert blocked.is_set()

    def test_wait_for_slot_closes(self, mock_cv: NonCallableMagicMock) -> None:
        q: deque[int] = deque()
        stream = Stream(0, q, mock_cv)

        with stream.wait_for_slot_or_cancelling() as slot:
            pass

        with pytest.raises(RuntimeError, match="closed"):
            if not slot:
                pass  # unexpected: stream is cancelling
            else:
                slot.put_result(42)

    def test_wait_for_slot_raises_after_finished(
        self, mock_cv: NonCallableMagicMock
    ) -> None:
        q: deque[int] = deque()
        stream = Stream(0, q, mock_cv)
        stream.finish(None)

        with (
            pytest.raises(RuntimeError, match="finished"),
            stream.wait_for_slot_or_cancelling() as _,
        ):
            pass

    @pytest.mark.parametrize(("maxsize", "n_items"), [(0, 0), (1, 0), (1, 1)])
    def test_wait_for_cancelling(
        self, maxsize: int, n_items: int, mock_cv: NonCallableMagicMock
    ) -> None:
        # regardless of `wait_for` being short-circuited by this mock, the
        # the result should reflect the _initial_ state (already cancelled)
        q: deque[int] = deque([1] * n_items)
        stream = Stream(maxsize, q, mock_cv)
        stream.cancel()

        with stream.wait_for_slot_or_cancelling() as slot:
            pass

        assert slot is None

    def test_wait_for_cancelling_would_block(self) -> None:
        q: deque[int] = deque([1])
        cv = CapturingCondition()
        stream = Stream(len(q), q, cv)

        blocked = Event()

        def on_block() -> None:
            blocked.set()
            stream.cancel()

        cv.call_on_next_block(on_block)

        with stream.wait_for_slot_or_cancelling() as slot:
            pass

        assert slot is None
        assert blocked.is_set()

    # NOTE: n_items = 0 tests finish alone
    @pytest.mark.parametrize("n_items", [0, 1, 2])
    def test_get_result_and_finish(
        self, n_items: int, mock_cv: NonCallableMagicMock
    ) -> None:
        stream: Stream[int] = Stream(0, deque(), mock_cv)

        items = list(range(n_items))
        for item in items:
            with stream.wait_for_slot_or_cancelling() as slot:
                if not slot:
                    break
                slot.put_result(item)
        stream.finish(None)

        items_recv: list[int] = []
        while True:
            item = stream.get_result()
            if isinstance(item, StreamSentinel):
                break
            items_recv.append(item)

        assert items == items_recv

    def test_finish_raises_after_finished(self, mock_cv: NonCallableMagicMock) -> None:
        stream: Stream[Never] = Stream(0, deque(), mock_cv)
        stream.finish(None)
        with pytest.raises(RuntimeError, match="finished"):
            stream.finish(None)

    def test_finish_forwards_exception(self, mock_cv: NonCallableMagicMock) -> None:
        stream: Stream[Never] = Stream(0, deque(), mock_cv)
        stream.finish(SentinelException())
        with pytest.raises(SentinelException):
            _ = stream.get_result()

    @pytest.mark.parametrize("n_items", [0, 1, 2])
    def test_get_result_would_block(self, n_items: int) -> None:
        q: deque[int] = deque()
        cv = CapturingCondition()
        stream = Stream(0, q, cv)
        items: list[int] = list(range(n_items))

        items_iter: Iterator[int] = iter(items)
        blocked = Event()

        def on_block() -> None:
            blocked.set()
            try:
                item = next(items_iter)
            except StopIteration:
                stream.finish(None)
            else:
                with stream.wait_for_slot_or_cancelling() as slot:
                    if not slot:
                        pass  # unexpected: stream cancelled
                    else:
                        slot.put_result(item)

        items_recv: list[int] = []
        had_blocked: list[bool] = []
        while True:
            cv.call_on_next_block(on_block)
            item = stream.get_result()
            if isinstance(item, StreamSentinel):
                break
            items_recv.append(item)
            had_blocked.append(blocked.is_set())
            blocked.clear()

        assert items_recv == items
        assert all(had_blocked)

    def test_cancel_would_block(self) -> None:
        cv = CapturingCondition()
        stream: Stream[int] = Stream(0, deque(), cv)

        blocked = Event()

        def on_block() -> None:
            blocked.set()
            stream.finish(None)

        cv.call_on_next_block(on_block)

        stream.cancel(block=True)

        assert blocked.is_set()


class TestIqueue:
    @pytest.fixture
    def items(self) -> list[int]:
        return list(range(10))

    def test_next(self, items: list[int]) -> None:
        items_iter = iter(items)

        items_recv: list[int] = []
        items_recv_iter = iqueue(items_iter)
        while True:
            try:
                item = next(items_recv_iter)
            except StopIteration:
                break
            else:
                items_recv.append(item)

        assert items_recv == items

    def test_next_raises(self) -> None:
        def raise_on_iter() -> Generator[Never, None, None]:
            raise SentinelException
            yield

        items_recv_iter = iqueue(raise_on_iter())

        with pytest.raises(SentinelException):
            _ = next(items_recv_iter)

    def test_for_loop(self, items: list[int]) -> None:
        items_recv: list[int] = []
        for item in iqueue(iter(items)):
            items_recv.append(item)  # noqa: PERF402  # explicitly tests for loop
        assert items_recv == items

    @pytest.mark.parametrize("n_items", [0, 1, 2])
    def test_for_loop_break_early(self, n_items: int) -> None:
        items_iter = iter(range(n_items + 2))

        items_recv: list[int] = []
        items_recv_iter = iqueue(items_iter)
        for i, item in enumerate(items_recv_iter):
            if i == n_items:
                break
            items_recv.append(item)
        next_item_recv = next(items_recv_iter)
        del items_recv_iter
        gc.collect()

        assert items_recv == list(range(n_items))
        assert next_item_recv == n_items + 1

    def test_list(self, items: list[int]) -> None:
        items_recv = list(iqueue(iter(items)))
        assert items_recv == items

    @pytest.mark.parametrize(
        ("close_action", "action"),
        [
            (_step_close, iter),
            (_step_close, next),
            (_step_context_manager, next),
        ],
    )
    def test_raises_on_access_after_close(
        self,
        close_action: Callable[[iqueue[Any]], Any],
        action: Callable[[iqueue[Any]], Any],
        items: list[int],
    ) -> None:
        items_iter = iter(items)
        items_recv_iter = iqueue(items_iter)
        close_action(items_recv_iter)
        with pytest.raises(RuntimeError, match="shut down"):
            _ = action(items_recv_iter)

    def test_exit_stack(self, items: list[int]) -> None:
        items_iter = iter(items)
        exit_stack = PipelineExitStack()
        with exit_stack:
            items_recv_iter = iqueue(items_iter)
        with pytest.raises(RuntimeError, match="shut down"):
            _ = next(items_recv_iter)

    def test_backpressure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cv = CapturingCondition()

        q = deque()
        monkeypatch.setattr(
            gpu_utils.Stream, "new", lambda maxsize=0: Stream(maxsize, q, cv)
        )

        consumer_wakeup = Event()
        producer_failed_to_block = Event()
        producer_wakeup = Event()
        done = Event()  # covered under cv

        def on_blocked_producer() -> None:
            consumer_wakeup.set()
            producer_wakeup.wait()

        def wait_for_failure_to_block() -> None:
            with cv:
                cv.wait_for(lambda: list(q) == [1, 2] or done.is_set(), private=True)
                if done.is_set():
                    return
            producer_failed_to_block.set()
            consumer_wakeup.set()

        items_recv_iter = iqueue(iter([1, 2]), maxsize=1)
        cv.call_on_next_block(on_blocked_producer, lambda: list(q) == [1])
        with ThreadPoolExecutor(2) as waiters:
            get_fut = waiters.submit(next, items_recv_iter)  # kick off iqueue
            block_fut = waiters.submit(wait_for_failure_to_block)
            try:
                consumer_wakeup.wait()
                if producer_failed_to_block.is_set():
                    pytest.fail("producer failed to block")
            finally:  # unblock waiter
                with cv:
                    done.set()
                    cv.notify_all()
                block_fut.result()

            # unblock producer
            _ = get_fut.result()
            producer_wakeup.set()
            _ = next(items_recv_iter)
            with pytest.raises(StopIteration):
                _ = next(items_recv_iter)


def test_pipeline_exit_stack_nested() -> None:
    items = range(10)
    with PipelineExitStack():
        iq = iqueue(iter(items))
        with PipelineExitStack():  # this exiting shouldn't break iq
            pass
        actual = list(iq)
    assert actual == list(items)
