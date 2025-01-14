import json

import pytest

from griptape.artifacts import TextArtifact
from griptape.structures import Agent
from griptape.tasks import ActionsSubtask, ToolTask
from tests.mocks.mock_prompt_driver import MockPromptDriver
from tests.mocks.mock_tool.tool import MockTool
from tests.utils import defaults


class TestToolTask:
    TARGET_TOOLS_SCHEMA = {
        "description": "JSON schema for an array of actions.",
        "type": "array",
        "items": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description: foo", "const": "test"},
                        "input": {
                            "type": "object",
                            "properties": {
                                "values": {
                                    "description": "Test input",
                                    "type": "object",
                                    "properties": {"test": {"type": "string"}},
                                    "required": ["test"],
                                    "additionalProperties": False,
                                }
                            },
                            "required": ["values"],
                            "additionalProperties": False,
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description", "const": "test_callable_schema"},
                        "input": {
                            "type": "object",
                            "properties": {
                                "values": {
                                    "description": "Test input",
                                    "type": "object",
                                    "properties": {"test": {"type": "string"}},
                                    "required": ["test"],
                                    "additionalProperties": False,
                                }
                            },
                            "required": ["values"],
                            "additionalProperties": False,
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description: foo", "const": "test_error"},
                        "input": {
                            "type": "object",
                            "properties": {
                                "values": {
                                    "description": "Test input",
                                    "type": "object",
                                    "properties": {"test": {"type": "string"}},
                                    "required": ["test"],
                                    "additionalProperties": False,
                                }
                            },
                            "required": ["values"],
                            "additionalProperties": False,
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description: foo", "const": "test_exception"},
                        "input": {
                            "type": "object",
                            "properties": {
                                "values": {
                                    "description": "Test input",
                                    "type": "object",
                                    "properties": {"test": {"type": "string"}},
                                    "required": ["test"],
                                    "additionalProperties": False,
                                }
                            },
                            "required": ["values"],
                            "additionalProperties": False,
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description", "const": "test_list_output"},
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                        "input": {"additionalProperties": False, "properties": {}, "required": [], "type": "object"},
                    },
                    "required": ["name", "path", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description", "const": "test_no_schema"},
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                        "input": {"additionalProperties": False, "properties": {}, "required": [], "type": "object"},
                    },
                    "required": ["name", "path", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description: foo", "const": "test_str_output"},
                        "input": {
                            "type": "object",
                            "properties": {
                                "values": {
                                    "description": "Test input",
                                    "type": "object",
                                    "properties": {"test": {"type": "string"}},
                                    "required": ["test"],
                                    "additionalProperties": False,
                                }
                            },
                            "required": ["values"],
                            "additionalProperties": False,
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description", "const": "test_without_default_memory"},
                        "input": {
                            "type": "object",
                            "properties": {
                                "values": {
                                    "description": "Test input",
                                    "type": "object",
                                    "properties": {"test": {"type": "string"}},
                                    "required": ["test"],
                                    "additionalProperties": False,
                                }
                            },
                            "required": ["values"],
                            "additionalProperties": False,
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
            ]
        },
        "$id": "Actions Schema",
        "$schema": "http://json-schema.org/draft-07/schema#",
    }

    @pytest.fixture()
    def agent(self, mock_config):
        output_dict = {"tag": "foo", "name": "MockTool", "path": "test", "input": {"values": {"test": "foobar"}}}

        mock_config.drivers_config.prompt_driver = MockPromptDriver(
            mock_output=f"```python foo bar\n{json.dumps(output_dict)}"
        )

        return Agent()

    def test_run_without_memory(self, agent):
        task = ToolTask(tool=MockTool())

        agent.add_task(task)

        assert task.run().name == "MockTool output"
        assert task.run().value == "ack foobar"

    def test_run_with_memory(self, agent):
        task = ToolTask(tool=MockTool(off_prompt=True))

        agent.add_task(task)

        assert task.run().to_text().startswith('Output of "MockTool.test" was stored in memory')

    def test_meta_memory(self):
        memory = defaults.text_task_memory("TestMemory")
        subtask = ActionsSubtask()
        agent = Agent(task_memory=memory)

        subtask.structure = agent

        memory.process_output(MockTool().test, subtask, TextArtifact("foo"))

        task = ToolTask(tool=MockTool())

        agent.add_task(task)

        system_template = task.generate_system_template(task)

        assert "You have access to additional contextual information" in system_template

    def test_actions_schema(self):
        tool = MockTool()
        task = ToolTask("test", tool=tool)

        Agent().add_task(task)

        assert task.actions_schema().json_schema("Actions Schema") == self.TARGET_TOOLS_SCHEMA

    def test_to_dict(self):
        tool = MockTool()
        task = ToolTask("test", tool=tool)

        expected_tool_task_dict = {
            "type": task.type,
            "id": task.id,
            "state": str(task.state),
            "parent_ids": task.parent_ids,
            "child_ids": task.child_ids,
            "max_meta_memory_entries": task.max_meta_memory_entries,
            "context": task.context,
            "rulesets": [],
            "rules": [],
            "prompt_driver": {
                "extra_params": {},
                "max_tokens": None,
                "stream": False,
                "temperature": 0.1,
                "type": "MockPromptDriver",
                "structured_output_strategy": "rule",
                "use_native_tools": False,
            },
            "tool": {
                "type": task.tool.type,
                "name": task.tool.name,
                "input_memory": task.tool.input_memory,
                "output_memory": task.tool.output_memory,
                "install_dependencies_on_init": task.tool.install_dependencies_on_init,
                "dependencies_install_directory": task.tool.dependencies_install_directory,
                "verbose": task.tool.verbose,
                "off_prompt": task.tool.off_prompt,
            },
        }
        assert expected_tool_task_dict == task.to_dict()

    def test_from_dict(self):
        tool = MockTool()
        task = ToolTask("test", tool=tool)

        serialized_tool_task = task.to_dict()
        serialized_tool_task["tool"]["module_name"] = "tests.mocks.mock_tool.tool"
        serialized_tool_task["prompt_driver"]["module_name"] = "tests.mocks.mock_prompt_driver"
        assert isinstance(serialized_tool_task, dict)

        deserialized_tool_task = ToolTask.from_dict(serialized_tool_task)
        assert isinstance(deserialized_tool_task, ToolTask)
