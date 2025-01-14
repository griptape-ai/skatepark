import pytest

from griptape.events import FinishStructureRunEvent, FinishTaskEvent, StartTaskEvent
from griptape.structures import Agent, Pipeline
from griptape.tasks import PromptTask
from tests.mocks.mock_prompt_driver import MockPromptDriver


class TestStructure:
    def test_output(self):
        pipeline = Pipeline()
        with pytest.raises(
            ValueError, match="Structure has no output Task. Add a Task to the Structure to generate output."
        ):
            assert pipeline.output

        agent = Agent()
        with pytest.raises(
            ValueError, match="Structure's output Task has no output. Run the Structure to generate output."
        ):
            assert agent.output

    def test_conversation_mode_per_structure(self):
        pipeline = Pipeline(
            conversation_memory_strategy="per_structure",
            tasks=[PromptTask("1"), PromptTask("2")],
        )

        pipeline.run()
        assert pipeline.conversation_memory is not None
        assert len(pipeline.conversation_memory.runs) == 1
        assert pipeline.conversation_memory.runs[0].input.value == "1"
        assert pipeline.conversation_memory.runs[0].output.value == "mock output"

    def test_conversation_mode_per_task(self):
        pipeline = Pipeline(
            conversation_memory_strategy="per_task",
            tasks=[PromptTask("1"), PromptTask("2")],
        )

        pipeline.run()
        assert pipeline.conversation_memory is not None
        assert len(pipeline.conversation_memory.runs) == 2
        assert pipeline.conversation_memory.runs[0].input.value == "1"
        assert pipeline.conversation_memory.runs[0].output.value == "mock output"
        assert pipeline.conversation_memory.runs[1].input.value == "2"
        assert pipeline.conversation_memory.runs[1].output.value == "mock output"

    def test_conversation_mode_per_task_no_memory(self):
        pipeline = Pipeline(
            conversation_memory_strategy="per_task",
            tasks=[PromptTask(conversation_memory=None), PromptTask("2")],
        )

        pipeline.run()
        assert pipeline.conversation_memory is not None
        assert len(pipeline.conversation_memory.runs) == 1
        assert pipeline.conversation_memory.runs[0].input.value == "2"

    def test_to_dict(self):
        task = PromptTask("test prompt")
        agent = Agent(prompt_driver=MockPromptDriver())
        agent.add_task(task)
        expected_agent_dict = {
            "type": "Agent",
            "id": agent.id,
            "tasks": [
                {
                    "type": agent.tasks[0].type,
                    "id": agent.tasks[0].id,
                    "state": str(agent.tasks[0].state),
                    "parent_ids": agent.tasks[0].parent_ids,
                    "child_ids": agent.tasks[0].child_ids,
                    "max_meta_memory_entries": agent.tasks[0].max_meta_memory_entries,
                    "context": agent.tasks[0].context,
                    "rulesets": [],
                    "rules": [],
                    "max_subtasks": 20,
                    "tools": [],
                    "prompt_driver": {
                        "extra_params": {},
                        "max_tokens": None,
                        "stream": False,
                        "temperature": 0.1,
                        "type": "MockPromptDriver",
                        "use_native_tools": False,
                        "structured_output_strategy": "rule",
                    },
                }
            ],
            "rulesets": [],
            "rules": [],
            "conversation_memory": {
                "type": agent.conversation_memory.type,
                "runs": agent.conversation_memory.runs,
                "meta": agent.conversation_memory.meta,
                "max_runs": agent.conversation_memory.max_runs,
            },
            "conversation_memory_strategy": agent.conversation_memory_strategy,
        }
        assert agent.to_dict() == expected_agent_dict

        agent.run()

        expected_agent_dict = {
            **expected_agent_dict,
            "tasks": [
                {
                    **expected_agent_dict["tasks"][0],
                    "state": str(agent.tasks[0].state),
                }
            ],
            "conversation_memory": {
                **expected_agent_dict["conversation_memory"],
                "runs": agent.conversation_memory.to_dict()["runs"],
            },
        }
        assert agent.to_dict() == expected_agent_dict

    def test_from_dict(self):
        task = PromptTask("test prompt")
        agent = Agent(prompt_driver=MockPromptDriver())
        agent.add_task(task)

        serialized_agent = agent.to_dict()
        assert isinstance(serialized_agent, dict)

        serialized_agent["tasks"][0]["prompt_driver"]["module_name"] = "tests.mocks.mock_prompt_driver"
        deserialized_agent = Agent.from_dict(serialized_agent)
        assert isinstance(deserialized_agent, Agent)

        assert deserialized_agent.task_outputs[task.id] is None
        deserialized_agent.run()

        assert len(deserialized_agent.task_outputs) == 1
        assert deserialized_agent.task_outputs[task.id].value == "mock output"

    def test_run_stream(self):
        from griptape.events import (
            EventBus,
            FinishPromptEvent,
            FinishStructureRunEvent,
            StartPromptEvent,
            StartStructureRunEvent,
        )

        agent = Agent()
        event_types = [
            StartStructureRunEvent,
            StartTaskEvent,
            StartPromptEvent,
            FinishPromptEvent,
            FinishTaskEvent,
            FinishStructureRunEvent,
        ]
        events = agent.run_stream()

        for idx, event in enumerate(events):
            assert isinstance(event, event_types[idx])
        assert len(EventBus.event_listeners) == 0

    def test_run_stream_custom_event_types(self):
        from griptape.events import EventBus, FinishPromptEvent, StartPromptEvent, StartStructureRunEvent

        agent = Agent()
        event_types = [StartStructureRunEvent, StartPromptEvent, FinishPromptEvent]
        expected_event_types = [StartStructureRunEvent, StartPromptEvent, FinishPromptEvent, FinishStructureRunEvent]
        events = agent.run_stream(event_types=event_types)

        for idx, event in enumerate(events):
            assert isinstance(event, expected_event_types[idx])
        assert len(EventBus.event_listeners) == 0
