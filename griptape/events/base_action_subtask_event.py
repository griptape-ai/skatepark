from __future__ import annotations
from attrs import define, field
from typing import Optional, TYPE_CHECKING
from .base_task_event import BaseTaskEvent


if TYPE_CHECKING:
    from griptape.tasks import ActionSubtask


@define
class BaseActionSubtaskEvent(BaseTaskEvent):
    subtask_parent_task_id: Optional[str] = field(kw_only=True)
    subtask_thought: Optional[str] = field(kw_only=True)
    subtask_action_type: Optional[str] = field(kw_only=True)
    subtask_action_name: Optional[str] = field(kw_only=True)
    subtask_action_input: Optional[dict] = field(kw_only=True)

    @classmethod
    def from_task(cls, task: ActionSubtask) -> BaseActionSubtaskEvent:
        return cls(
            task_id=task.id,
            task_parent_ids=task.parent_ids,
            task_child_ids=task.child_ids,
            task_input=task.input,
            task_output=task.output,
            subtask_parent_task_id=task.parent_task_id,
            subtask_thought=task.thought,
            subtask_action_type=task.action_type,
            subtask_action_name=task.action_name,
            subtask_action_input=task.action_input,
        )
