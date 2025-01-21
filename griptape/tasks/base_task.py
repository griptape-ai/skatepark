from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from attrs import Factory, define, field

from griptape.artifacts import BaseArtifact, ErrorArtifact
from griptape.configs import Defaults
from griptape.events import EventBus, FinishTaskEvent, StartTaskEvent
from griptape.mixins.futures_executor_mixin import FuturesExecutorMixin
from griptape.mixins.runnable_mixin import RunnableMixin
from griptape.mixins.serializable_mixin import SerializableMixin

if TYPE_CHECKING:
    from griptape.memory.meta import BaseMetaEntry
    from griptape.structures import Structure

logger = logging.getLogger(Defaults.logging_config.logger_name)


@define
class BaseTask(FuturesExecutorMixin, SerializableMixin, RunnableMixin["BaseTask"], ABC):
    class State(Enum):
        PENDING = 1
        RUNNING = 2
        FINISHED = 3
        SKIPPED = 4

    id: str = field(default=Factory(lambda: uuid.uuid4().hex), kw_only=True, metadata={"serializable": True})
    state: State = field(default=State.PENDING, kw_only=True, metadata={"serializable": True})
    parent_ids: list[str] = field(factory=list, kw_only=True, metadata={"serializable": True})
    child_ids: list[str] = field(factory=list, kw_only=True, metadata={"serializable": True})
    max_meta_memory_entries: Optional[int] = field(default=20, kw_only=True, metadata={"serializable": True})
    structure: Optional[Structure] = field(default=None, kw_only=True)

    output: Optional[BaseArtifact] = field(default=None, init=False)
    context: dict[str, Any] = field(factory=dict, kw_only=True, metadata={"serializable": True})
    _execution_args: tuple = field(factory=tuple, init=False)
    _execution_kwargs: dict[str, Any] = field(factory=dict, init=False)

    @property
    def execution_args(self) -> tuple:
        return self._execution_args

    @property
    def execution_kwargs(self) -> dict:
        return self._execution_kwargs

    def __rshift__(self, other: BaseTask | list[BaseTask]) -> BaseTask | list[BaseTask]:
        if isinstance(other, list):
            self.add_children(other)
        else:
            self.add_child(other)

        return other

    def __lshift__(self, other: BaseTask | list[BaseTask]) -> BaseTask | list[BaseTask]:
        if isinstance(other, list):
            self.add_parents(other)
        else:
            self.add_parent(other)

        return other

    def __attrs_post_init__(self) -> None:
        if self.structure is not None:
            self.structure.add_task(self)

    @property
    @abstractmethod
    def input(self) -> BaseArtifact: ...

    @property
    def parents(self) -> list[BaseTask]:
        if self.structure is not None:
            return [self.structure.find_task(parent_id) for parent_id in self.parent_ids]
        raise ValueError("Structure must be set to access parents")

    @property
    def children(self) -> list[BaseTask]:
        if self.structure is not None:
            return [self.structure.find_task(child_id) for child_id in self.child_ids]
        raise ValueError("Structure must be set to access children")

    @property
    def parent_outputs(self) -> dict[str, BaseArtifact]:
        return {parent.id: parent.output for parent in self.parents if parent.output}

    @property
    def parents_output_text(self) -> str:
        return "\n".join([parent.output.to_text() for parent in self.parents if parent.output])

    @property
    def meta_memories(self) -> list[BaseMetaEntry]:
        if self.structure is not None and self.structure.meta_memory:
            if self.max_meta_memory_entries:
                return self.structure.meta_memory.entries[: self.max_meta_memory_entries]
            else:
                return self.structure.meta_memory.entries
        else:
            return []

    def __str__(self) -> str:
        return str(self.output.value) if self.output is not None else ""

    def add_parents(self, parents: list[BaseTask]) -> None:
        for parent in parents:
            self.add_parent(parent)

    def add_parent(self, parent: BaseTask) -> BaseTask:
        if parent.id not in self.parent_ids:
            self.parent_ids.append(parent.id)

        if self.id not in parent.child_ids:
            parent.child_ids.append(self.id)

        if self.structure is not None and parent not in self.structure.tasks:
            self.structure.add_task(parent)

        return self

    def add_children(self, children: list[BaseTask]) -> None:
        for child in children:
            self.add_child(child)

    def add_child(self, child: BaseTask) -> BaseTask:
        if child.id not in self.child_ids:
            self.child_ids.append(child.id)

        if self.id not in child.parent_ids:
            child.parent_ids.append(self.id)

        if self.structure is not None and child not in self.structure.tasks:
            self.structure.add_task(child)

        return self

    def preprocess(self, structure: Structure) -> BaseTask:
        self.structure = structure

        return self

    def is_pending(self) -> bool:
        return self.state == BaseTask.State.PENDING

    def is_finished(self) -> bool:
        return self.state == BaseTask.State.FINISHED

    def is_running(self) -> bool:
        return self.state == BaseTask.State.RUNNING

    def is_skipped(self) -> bool:
        return self.state == BaseTask.State.SKIPPED

    def before_run(self) -> None:
        super().before_run()
        if self.structure is not None:
            EventBus.publish_event(
                StartTaskEvent(
                    task_id=self.id,
                    task_parent_ids=self.parent_ids,
                    task_child_ids=self.child_ids,
                    task_input=self.input,
                    task_output=self.output,
                ),
            )

    def run(self, *args, **kwargs) -> BaseArtifact:
        try:
            self._execution_args = args
            self._execution_kwargs = kwargs

            self.state = BaseTask.State.RUNNING

            self.before_run()

            self.output = self.try_run()

            self.after_run()
        except Exception as e:
            logger.exception("%s %s\n%s", self.__class__.__name__, self.id, e)

            self.output = ErrorArtifact(str(e), exception=e)
        finally:
            self.state = BaseTask.State.FINISHED

        return self.output

    def after_run(self) -> None:
        super().after_run()
        if self.structure is not None:
            EventBus.publish_event(
                FinishTaskEvent(
                    task_id=self.id,
                    task_parent_ids=self.parent_ids,
                    task_child_ids=self.child_ids,
                    task_input=self.input,
                    task_output=self.output,
                ),
            )

    def can_run(self) -> bool:
        # If this Task has been skipped or is not pending, it should not run
        if self.is_skipped() or not self.is_pending():
            return False

        # If this Task has parents, and _all_ of them are skipped, it should not run
        if self.parents and all(parent.is_skipped() for parent in self.parents):
            self.state = BaseTask.State.SKIPPED
            return False

        # If _all_ this Task's unskipped parents are finished, it should run
        unskipped_parents = [parent for parent in self.parents if not parent.is_skipped()]

        return all(parent.is_finished() for parent in unskipped_parents)

    def reset(self) -> BaseTask:
        self.state = BaseTask.State.PENDING
        self.output = None
        self._execution_args = ()
        self._execution_kwargs = {}

        return self

    @abstractmethod
    def try_run(self) -> BaseArtifact: ...

    @property
    def full_context(self) -> dict[str, Any]:
        # Need to deep copy so that the serialized context doesn't contain non-serializable data
        context = deepcopy(self.context)
        if self.structure is None:
            context.update({"args": self._execution_args, "kwargs": self._execution_kwargs})
        else:
            context.update(self.structure.context(self))

        return context
