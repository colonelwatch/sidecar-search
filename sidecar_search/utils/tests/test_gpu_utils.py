from concurrent.futures import Future
from typing import Generator, Iterator, Never

import pytest

from sidecar_search.utils.gpu_utils import consume_futures


class TestConsumeFutures:
    def test_blocking(self) -> None:
        fut: Future[Never] = Future()
        futs = iter((fut,))
        with pytest.raises(TimeoutError):
            _ = list(consume_futures(futs, 0, yield_timeout=0))

    def test_blocking_after_exhaustion(self) -> None:
        fut: Future[Never] = Future()
        futs = iter((fut,))
        with pytest.raises(TimeoutError):
            _ = list(consume_futures(futs, 10, yield_timeout=0))

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
