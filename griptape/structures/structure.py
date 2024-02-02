from __future__ import annotations

import logging
import uuid
import warnings
from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING, Any, Optional

from attrs import Factory, define, field
from rich.logging import RichHandler

from griptape.artifacts import BlobArtifact, TextArtifact
from griptape.config import BaseStructureConfig, OpenAiStructureConfig
from griptape.drivers import BaseEmbeddingDriver, BasePromptDriver
from griptape.engines import CsvExtractionEngine, JsonExtractionEngine, PromptSummaryEngine, VectorQueryEngine
from griptape.events import BaseEvent, EventListener
from griptape.events.finish_structure_run_event import FinishStructureRunEvent
from griptape.events.start_structure_run_event import StartStructureRunEvent
from griptape.memory import TaskMemory
from griptape.memory.meta import MetaMemory
from griptape.memory.structure import ConversationMemory
from griptape.memory.task.storage import BlobArtifactStorage, TextArtifactStorage
from griptape.rules import Rule, Ruleset
from griptape.tasks import BaseTask

if TYPE_CHECKING:
    from griptape.memory.structure import BaseConversationMemory


@define
class Structure(ABC):
    LOGGER_NAME = "griptape"

    id: str = field(default=Factory(lambda: uuid.uuid4().hex), kw_only=True)
    stream: Optional[bool] = field(default=None, kw_only=True)
    prompt_driver: Optional[BasePromptDriver] = field(default=None)
    embedding_driver: Optional[BaseEmbeddingDriver] = field(default=None, kw_only=True)
    config: BaseStructureConfig = field(
        default=Factory(lambda self: self.default_config, takes_self=True), kw_only=True
    )
    rulesets: list[Ruleset] = field(factory=list, kw_only=True)
    rules: list[Rule] = field(factory=list, kw_only=True)
    tasks: list[BaseTask] = field(factory=list, kw_only=True)
    custom_logger: Optional[Logger] = field(default=None, kw_only=True)
    logger_level: int = field(default=logging.INFO, kw_only=True)
    event_listeners: list[EventListener] = field(factory=list, kw_only=True)
    conversation_memory: BaseConversationMemory = field(
        default=Factory(
            lambda self: ConversationMemory(driver=self.config.global_drivers.conversation_memory_driver),
            takes_self=True,
        ),
        kw_only=True,
    )
    task_memory: Optional[TaskMemory] = field(
        default=Factory(lambda self: self.default_task_memory, takes_self=True), kw_only=True
    )
    meta_memory: MetaMemory = field(default=Factory(lambda: MetaMemory()), kw_only=True)
    _execution_args: tuple = ()
    _logger: Optional[Logger] = None

    @rulesets.validator  # pyright: ignore
    def validate_rulesets(self, _, rulesets: list[Ruleset]) -> None:
        if not rulesets:
            return

        if self.rules:
            raise ValueError("can't have both rulesets and rules specified")

    @rules.validator  # pyright: ignore
    def validate_rules(self, _, rules: list[Rule]) -> None:
        if not rules:
            return

        if self.rulesets:
            raise ValueError("can't have both rules and rulesets specified")

    def __attrs_post_init__(self) -> None:
        if self.conversation_memory:
            self.conversation_memory.structure = self

        tasks = self.tasks.copy()
        self.tasks.clear()
        self.add_tasks(*tasks)

    def __add__(self, other: BaseTask | list[BaseTask]) -> list[BaseTask]:
        return self.add_tasks(*other) if isinstance(other, list) else self + [other]

    @property
    def execution_args(self) -> tuple:
        return self._execution_args

    @property
    def logger(self) -> Logger:
        if self.custom_logger:
            return self.custom_logger
        else:
            if self._logger is None:
                self._logger = logging.getLogger(self.LOGGER_NAME)

                self._logger.propagate = False
                self._logger.level = self.logger_level

                self._logger.handlers = [RichHandler(show_time=True, show_path=False)]
            return self._logger

    @property
    def input_task(self) -> Optional[BaseTask]:
        return self.tasks[0] if self.tasks else None

    @property
    def output_task(self) -> Optional[BaseTask]:
        return self.tasks[-1] if self.tasks else None

    @property
    def finished_tasks(self) -> list[BaseTask]:
        return [s for s in self.tasks if s.is_finished()]

    @property
    def default_config(self) -> BaseStructureConfig:
        config = OpenAiStructureConfig()

        if self.prompt_driver is not None:
            warnings.warn(
                "`prompt_driver` is deprecated, use `config.global_drivers.prompt_driver` instead.", DeprecationWarning
            )
            config.global_drivers.prompt_driver = self.prompt_driver
            config.task_memory.query_engine.prompt_driver = self.prompt_driver
            config.task_memory.summary_engine.prompt_driver = self.prompt_driver
            config.task_memory.extraction_engine.csv.prompt_driver = self.prompt_driver
            config.task_memory.extraction_engine.json.prompt_driver = self.prompt_driver
        if self.embedding_driver is not None:
            warnings.warn(
                "`embedding_driver` is deprecated, use `config.global_drivers.embedding_driver` instead.",
                DeprecationWarning,
            )
            config.task_memory.query_engine.vector_store_driver.embedding_driver = self.embedding_driver
        if self.stream is not None:
            warnings.warn("`stream` is deprecated, use `config.prompt_driver.stream` instead.", DeprecationWarning)
            config.global_drivers.prompt_driver.stream = self.stream

        return config

    @property
    def default_task_memory(self) -> TaskMemory:
        global_drivers = self.config.global_drivers
        task_memory = self.config.task_memory
        return TaskMemory(
            artifact_storages={
                TextArtifact: TextArtifactStorage(
                    query_engine=VectorQueryEngine(
                        prompt_driver=task_memory.query_engine.prompt_driver or global_drivers.prompt_driver,
                        vector_store_driver=task_memory.query_engine.vector_store_driver
                        or global_drivers.vector_store_driver,
                    ),
                    summary_engine=PromptSummaryEngine(
                        prompt_driver=task_memory.summary_engine.prompt_driver or global_drivers.prompt_driver
                    ),
                    csv_extraction_engine=CsvExtractionEngine(
                        prompt_driver=task_memory.extraction_engine.csv.prompt_driver or global_drivers.prompt_driver
                    ),
                    json_extraction_engine=JsonExtractionEngine(
                        prompt_driver=task_memory.extraction_engine.json.prompt_driver or global_drivers.prompt_driver
                    ),
                ),
                BlobArtifact: BlobArtifactStorage(),
            }
        )

    def is_finished(self) -> bool:
        return all(s.is_finished() for s in self.tasks)

    def is_executing(self) -> bool:
        return any(s for s in self.tasks if s.is_executing())

    def find_task(self, task_id: str) -> BaseTask:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"Task with id {task_id} doesn't exist.")

    def add_tasks(self, *tasks: BaseTask) -> list[BaseTask]:
        return [self.add_task(s) for s in tasks]

    def add_event_listener(self, event_listener: EventListener) -> EventListener:
        if event_listener not in self.event_listeners:
            self.event_listeners.append(event_listener)

        return event_listener

    def remove_event_listener(self, event_listener: EventListener) -> None:
        if event_listener in self.event_listeners:
            self.event_listeners.remove(event_listener)
        else:
            raise ValueError("Event Listener not found.")

    def publish_event(self, event: BaseEvent) -> None:
        for event_listener in self.event_listeners:
            handler = event_listener.handler
            event_types = event_listener.event_types

            if event_types is None or type(event) in event_types:
                handler(event)

    def context(self, task: BaseTask) -> dict[str, Any]:
        return {"args": self.execution_args, "structure": self}

    def before_run(self) -> None:
        self.publish_event(StartStructureRunEvent())

    def after_run(self) -> None:
        self.publish_event(FinishStructureRunEvent())

    @abstractmethod
    def add_task(self, task: BaseTask) -> BaseTask:
        ...

    def run(self, *args) -> Structure:
        self.before_run()

        result = self.try_run(*args)

        self.after_run()

        return result

    @abstractmethod
    def try_run(self, *args) -> Structure:
        ...
