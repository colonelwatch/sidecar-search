import gc
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import count, cycle
from threading import Condition, Event
from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    Never,
    TypedDict,
    cast,
)
from unittest.mock import ANY, MagicMock, NonCallableMagicMock, call, create_autospec

import pytest
import torch

from .. import gpu_utils
from ..gpu_utils import (
    Stream,
    StreamSentinel,
    StreamSlot,
    consume_futures,
    imap,
    imap_multi_gpu,
    iqueue,
)
from .capturing_condition import CapturingCondition


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
    monkeypatch.setattr(gpu_utils, "imap", mock_imap)
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

        with pytest.raises(RuntimeError, match="finished"):
            with stream.wait_for_slot_or_cancelling() as _:
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
            items_recv.append(item)
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

    def test_context_manager(self, items: list[int]) -> None:
        items_iter = iter(items)
        with iqueue(items_iter) as items_recv_iter:
            items_recv = list(items_recv_iter)
        assert items_recv == items

    @pytest.mark.parametrize("n_items", [0, 1, 2])
    def test_context_manager_break_early(self, n_items: int) -> None:
        items_iter = iter(range(n_items + 2))
        items_recv: list[int] = []

        with iqueue(items_iter) as items_recv_iter:
            for i, item_recv in enumerate(items_recv_iter):
                if i == n_items:
                    break
                items_recv.append(item_recv)
            next_item_recv = next(items_recv_iter)

        assert items_recv == list(range(n_items))
        assert next_item_recv == n_items + 1

    def test_context_manager_raises_on_second_use(self, items: list[int]) -> None:
        items_iter = iter(items)
        with iqueue(items_iter) as items_recv_iter:
            pass
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

        cv.call_on_next_block(on_blocked_producer, lambda: list(q) == [1])
        with (
            ThreadPoolExecutor(1) as waiter,
            iqueue(iter([1, 2]), maxsize=1) as items_recv_iter,
        ):
            fut = waiter.submit(wait_for_failure_to_block)
            try:
                consumer_wakeup.wait()
                if producer_failed_to_block.is_set():
                    pytest.fail("producer failed to block")
            finally:  # unblock waiter
                with cv:
                    done.set()
                    cv.notify_all()
                fut.result()

            # unblock producer
            _ = next(items_recv_iter)
            producer_wakeup.set()
            _ = next(items_recv_iter)
            with pytest.raises(StopIteration):
                _ = next(items_recv_iter)
