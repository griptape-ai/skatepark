from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape import utils
from griptape.artifacts import BaseArtifact, ErrorArtifact, InfoArtifact, ListArtifact
from griptape.mixins.actions_subtask_origin_mixin import ActionsSubtaskOriginMixin
from griptape.tasks import ActionsSubtask, PromptTask
from griptape.utils import J2

if TYPE_CHECKING:
    from schema import Schema

    from griptape.common import PromptStack
    from griptape.memory import TaskMemory
    from griptape.structures import Structure
    from griptape.tools import BaseTool


@define
class ToolTask(PromptTask, ActionsSubtaskOriginMixin):
    DEFAULT_MAX_STEPS = 0
    ACTION_PATTERN = r"(?s)[^{]*({.*})"

    tool: BaseTool = field(kw_only=True, metadata={"serializable": True})
    subtask: Optional[ActionsSubtask] = field(default=None, kw_only=True)
    task_memory: Optional[TaskMemory] = field(default=None, kw_only=True)
    tools: list[BaseTool] = field(factory=list, kw_only=True, metadata={"serializable": False})
    max_subtasks: int = field(default=DEFAULT_MAX_STEPS, kw_only=True, metadata={"serializable": False})

    @property
    def prompt_stack(self) -> PromptStack:
        stack = super().prompt_stack
        stack.tools = [self.tool]

        return stack

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.task_memory is not None:
            self.set_default_tools_memory(self.task_memory)

    def preprocess(self, structure: Structure) -> ToolTask:
        super().preprocess(structure)

        if self.task_memory is None and structure.task_memory is not None:
            self.set_default_tools_memory(structure.task_memory)

        return self

    def default_generate_system_template(self, _: PromptTask) -> str:
        return J2("tasks/tool_task/system.j2").render(
            rulesets=J2("rulesets/rulesets.j2").render(rulesets=self.all_rulesets),
            action_schema=utils.minify_json(json.dumps(self.tool.schema())),
            meta_memory=J2("memory/meta/meta_memory.j2").render(meta_memories=self.meta_memories),
            use_native_tools=self.prompt_driver.use_native_tools,
        )

    def actions_schema(self) -> Schema:
        return self._actions_schema_for_tools([self.tool])

    def try_run(self) -> BaseArtifact:
        result = self.prompt_driver.run(self.prompt_stack)

        if self.prompt_driver.use_native_tools:
            subtask_input = result.to_artifact()
        else:
            action_matches = re.findall(self.ACTION_PATTERN, result.to_text(), re.DOTALL)

            if not action_matches:
                return ErrorArtifact("No action found in prompt output.")
            data = action_matches[-1]
            action_dict = json.loads(data)

            action_dict["tag"] = self.tool.name
            subtask_input = J2("tasks/tool_task/subtask.j2").render(action_json=json.dumps(action_dict))

        try:
            subtask = self.add_subtask(ActionsSubtask(subtask_input))

            subtask.run()

            if isinstance(subtask.output, ListArtifact):
                first_artifact = subtask.output[0]
                if isinstance(first_artifact, BaseArtifact):
                    self.output = first_artifact
                else:
                    raise ValueError(f"Output is not an Artifact: {type(first_artifact)}")
            else:
                self.output = InfoArtifact("No tool output")
        except Exception as e:
            self.output = ErrorArtifact(f"Error processing tool input: {e}", exception=e)
        return self.output

    def find_tool(self, tool_name: str) -> BaseTool:
        if self.tool.name == tool_name:
            return self.tool
        else:
            raise ValueError(f"Tool with name {tool_name} not found.")

    def find_memory(self, memory_name: str) -> TaskMemory:
        raise NotImplementedError("ToolTask does not support Task Memory.")

    def find_subtask(self, subtask_id: str) -> ActionsSubtask:
        if self.subtask and self.subtask.id == subtask_id:
            return self.subtask
        else:
            raise ValueError(f"Subtask with id {subtask_id} not found.")

    def add_subtask(self, subtask: ActionsSubtask) -> ActionsSubtask:
        self.subtask = subtask
        self.subtask.attach_to(self)

        return self.subtask

    def set_default_tools_memory(self, memory: TaskMemory) -> None:
        self.task_memory = memory

        if self.task_memory:
            if self.tool.input_memory is None:
                self.tool.input_memory = [self.task_memory]
            if self.tool.output_memory is None and self.tool.off_prompt:
                self.tool.output_memory = {getattr(a, "name"): [self.task_memory] for a in self.tool.activities()}
