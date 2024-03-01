from __future__ import annotations
from abc import ABC
from attrs import define, field
from typing import TYPE_CHECKING, Optional
from .base_task_event import BaseTaskEvent


if TYPE_CHECKING:
    from griptape.tasks import BaseTask, ActionsSubtask


@define
class BaseActionSubtaskEvent(BaseTaskEvent, ABC):
    subtask_parent_task_id: Optional[str] = field(kw_only=True, metadata={"serializable": True})
    subtask_thought: Optional[str] = field(kw_only=True, metadata={"serializable": True})
    subtask_actions: Optional[list[dict]] = field(kw_only=True, metadata={"serializable": True})

    @classmethod
    def from_task(cls, task: BaseTask) -> BaseActionSubtaskEvent:
        if not isinstance(task, ActionsSubtask):
            raise ValueError("Event must be of instance ActionSubtask.")
        return cls(
            task_id=task.id,
            task_parent_ids=task.parent_ids,
            task_child_ids=task.child_ids,
            task_input=task.input,
            task_output=task.output,
            subtask_parent_task_id=task.parent_task_id,
            subtask_thought=task.thought,
            subtask_actions=task.actions
        )
