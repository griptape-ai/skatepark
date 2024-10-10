from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from attrs import Factory, define, field

from griptape.mixins.exponential_backoff_mixin import ExponentialBackoffMixin
from griptape.mixins.futures_executor_mixin import FuturesExecutorMixin

if TYPE_CHECKING:
    from griptape.events import BaseEvent

logger = logging.getLogger(__name__)


@define
class BaseEventListenerDriver(FuturesExecutorMixin, ExponentialBackoffMixin, ABC):
    batched: bool = field(default=True, kw_only=True)
    batch_size: int = field(default=10, kw_only=True)

    _batch: list[dict] = field(default=Factory(list), kw_only=True)

    @property
    def batch(self) -> list[dict]:
        return self._batch

    def publish_event(self, event: BaseEvent | dict) -> None:
        event_payload = event if isinstance(event, dict) else event.to_dict()

        if self.batched:
            self._batch.append(event_payload)
            if len(self.batch) >= self.batch_size:
                self.futures_executor.submit(self._safe_publish_event_payload_batch, self.batch)
                self._batch = []
        else:
            self.futures_executor.submit(self._safe_publish_event_payload, event_payload)

    def flush_events(self) -> None:
        if self.batch:
            self.futures_executor.submit(self._safe_publish_event_payload_batch, self.batch)
            self._batch = []

    @abstractmethod
    def try_publish_event_payload(self, event_payload: dict) -> None: ...

    @abstractmethod
    def try_publish_event_payload_batch(self, event_payload_batch: list[dict]) -> None: ...

    def _safe_publish_event_payload(self, event_payload: dict) -> None:
        for attempt in self.retrying():
            with attempt:
                self.try_publish_event_payload(event_payload)
        else:
            logger.error("event listener driver failed after all retry attempts")

    def _safe_publish_event_payload_batch(self, event_payload_batch: list[dict]) -> None:
        for attempt in self.retrying():
            with attempt:
                self.try_publish_event_payload_batch(event_payload_batch)
        else:
            logger.error("event listener driver failed after all retry attempts")
