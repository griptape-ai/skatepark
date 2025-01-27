from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from attrs import Factory, define, field

from griptape.common import observable
from griptape.events import EventBus, FinishStructureRunEvent, StartStructureRunEvent
from griptape.events.base_event import BaseEvent
from griptape.events.event_listener import EventListener
from griptape.memory import TaskMemory
from griptape.memory.meta import MetaMemory
from griptape.memory.structure import ConversationMemory, Run
from griptape.mixins.rule_mixin import RuleMixin
from griptape.mixins.runnable_mixin import RunnableMixin
from griptape.mixins.serializable_mixin import SerializableMixin
from griptape.utils.contextvars_utils import with_contextvars

if TYPE_CHECKING:
    from collections.abc import Iterator

    from griptape.artifacts import BaseArtifact
    from griptape.memory.structure import BaseConversationMemory
    from griptape.tasks import BaseTask


@define
class Structure(RuleMixin, SerializableMixin, RunnableMixin["Structure"], ABC):
    id: str = field(default=Factory(lambda: uuid.uuid4().hex), kw_only=True, metadata={"serializable": True})
    _tasks: list[Union[BaseTask, list[BaseTask]]] = field(
        factory=list, kw_only=True, alias="tasks", metadata={"serializable": True}
    )
    conversation_memory: Optional[BaseConversationMemory] = field(
        default=Factory(lambda: ConversationMemory()),
        kw_only=True,
        metadata={"serializable": True},
    )
    conversation_memory_strategy: Literal["per_structure", "per_task"] = field(
        default="per_structure", kw_only=True, metadata={"serializable": True}
    )
    task_memory: TaskMemory = field(
        default=Factory(lambda self: TaskMemory(), takes_self=True),
        kw_only=True,
    )
    meta_memory: MetaMemory = field(default=Factory(lambda: MetaMemory()), kw_only=True)
    fail_fast: bool = field(default=True, kw_only=True, metadata={"serializable": True})
    _execution_args: tuple = field(factory=tuple, init=False)
    _execution_kwargs: dict[str, Any] = field(factory=dict, init=False)
    _event_queue: Queue[BaseEvent] = field(default=Factory(lambda: Queue()), init=False)

    def __attrs_post_init__(self) -> None:
        tasks = self._tasks.copy()
        self._tasks.clear()
        self.add_tasks(*tasks)

    def __add__(self, other: BaseTask | list[BaseTask | list[BaseTask]]) -> list[BaseTask]:
        return self.add_tasks(*other) if isinstance(other, list) else self.add_tasks(other)

    @property
    def tasks(self) -> list[BaseTask]:
        tasks = []

        for task in self._tasks:
            if isinstance(task, list):
                tasks.extend(task)
            else:
                tasks.append(task)
        return tasks

    @property
    def execution_args(self) -> tuple:
        return self._execution_args

    @property
    def execution_kwargs(self) -> dict:
        return self._execution_kwargs

    @property
    def input_task(self) -> Optional[BaseTask]:
        return self.tasks[0] if self.tasks else None

    @property
    def output_task(self) -> Optional[BaseTask]:
        return self.tasks[-1] if self.tasks else None

    @property
    def output(self) -> BaseArtifact:
        if self.output_task is None:
            raise ValueError("Structure has no output Task. Add a Task to the Structure to generate output.")
        if self.output_task.output is None:
            raise ValueError("Structure's output Task has no output. Run the Structure to generate output.")
        return self.output_task.output

    @property
    def task_outputs(self) -> dict[str, Optional[BaseArtifact]]:
        return {task.id: task.output for task in self.tasks}

    @property
    def finished_tasks(self) -> list[BaseTask]:
        return [s for s in self.tasks if s.is_finished()]

    def is_finished(self) -> bool:
        return all(not s.can_run() for s in self.tasks)

    def is_running(self) -> bool:
        return any(s for s in self.tasks if s.is_running())

    def find_task(self, task_id: str) -> BaseTask:
        if (task := self.try_find_task(task_id)) is not None:
            return task
        raise ValueError(f"Task with id {task_id} doesn't exist.")

    def try_find_task(self, task_id: str) -> Optional[BaseTask]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def add_tasks(self, *tasks: BaseTask | list[BaseTask]) -> list[BaseTask]:
        added_tasks = []
        for task in tasks:
            if isinstance(task, list):
                added_tasks.extend(self.add_tasks(*task))
            else:
                added_tasks.append(self.add_task(task))
        return added_tasks

    def context(self, task: BaseTask) -> dict[str, Any]:
        return {"args": self.execution_args, "kwargs": self.execution_kwargs, "structure": self}

    def resolve_relationships(self) -> None:
        task_by_id = {}
        for task in self.tasks:
            if task.id in task_by_id:
                raise ValueError(f"Duplicate task with id {task.id} found.")
            task_by_id[task.id] = task

        for task in self.tasks:
            # Ensure parents include this task as a child
            for parent_id in task.parent_ids:
                if parent_id not in task_by_id:
                    raise ValueError(f"Task with id {parent_id} doesn't exist.")
                parent = task_by_id[parent_id]
                if task.id not in parent.child_ids:
                    parent.child_ids.append(task.id)

            # Ensure children include this task as a parent
            for child_id in task.child_ids:
                if child_id not in task_by_id:
                    raise ValueError(f"Task with id {child_id} doesn't exist.")
                child = task_by_id[child_id]
                if task.id not in child.parent_ids:
                    child.parent_ids.append(task.id)

    @observable
    def before_run(self, args: Any) -> None:
        super().before_run(args)

        [task.reset() for task in self.tasks]

        if self.input_task is not None:
            EventBus.publish_event(
                StartStructureRunEvent(
                    structure_id=self.id,
                    input_task_input=self.input_task.input,
                    input_task_output=self.input_task.output,
                ),
            )

        self.resolve_relationships()

    @observable
    def after_run(self) -> None:
        super().after_run()

        if self.output_task is not None:
            if (
                self.conversation_memory_strategy == "per_structure"
                and self.conversation_memory is not None
                and self.input_task is not None
                and self.output_task.output is not None
            ):
                run = Run(input=self.input_task.input, output=self.output_task.output)

                self.conversation_memory.add_run(run)

            EventBus.publish_event(
                FinishStructureRunEvent(
                    structure_id=self.id,
                    output_task_input=self.output_task.input,
                    output_task_output=self.output_task.output,
                ),
                flush=True,
            )

    @abstractmethod
    def add_task(self, task: BaseTask) -> BaseTask: ...

    @observable
    def run(self, *args, **kwargs) -> Structure:
        self._execution_args = args
        self._execution_kwargs = kwargs
        self.before_run(args)

        result = self.try_run(*args)

        self.after_run()

        return result

    @observable
    def run_stream(self, *args, event_types: Optional[list[type[BaseEvent]]] = None) -> Iterator[BaseEvent]:
        if event_types is None:
            event_types = [BaseEvent]
        else:
            if FinishStructureRunEvent not in event_types:
                event_types = [*event_types, FinishStructureRunEvent]

        with EventListener(self._event_queue.put, event_types=event_types):
            t = Thread(target=with_contextvars(self.run), args=args)
            t.start()

            while True:
                event = self._event_queue.get()
                if isinstance(event, FinishStructureRunEvent):
                    break
                else:
                    yield event
            t.join()

    @abstractmethod
    def try_run(self, *args) -> Structure: ...
