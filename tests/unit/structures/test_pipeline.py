import time

import pytest

from griptape.artifacts import ErrorArtifact, TextArtifact
from griptape.memory.structure import ConversationMemory
from griptape.rules import Rule, Ruleset
from griptape.structures import Pipeline
from griptape.tasks import BaseTask, CodeExecutionTask, PromptTask
from griptape.tokenizers import OpenAiTokenizer
from tests.mocks.mock_tool.tool import MockTool


class TestPipeline:
    @pytest.fixture()
    def waiting_task(self):
        def fn(task):
            time.sleep(2)
            return TextArtifact("done")

        return CodeExecutionTask(on_run=fn)

    @pytest.fixture()
    def error_artifact_task(self):
        def fn(task):
            return ErrorArtifact("error")

        return CodeExecutionTask(on_run=fn)

    def test_init(self):
        pipeline = Pipeline(rulesets=[Ruleset("TestRuleset", [Rule("test")])])

        assert pipeline.input_task is None
        assert pipeline.output_task is None
        assert pipeline.rulesets[0].name == "TestRuleset"
        assert pipeline.rulesets[0].rules[0].value == "test"
        assert pipeline.conversation_memory is not None

    def test_rulesets(self):
        pipeline = Pipeline(rulesets=[Ruleset("Foo", [Rule("foo test")])])

        pipeline.add_tasks(
            PromptTask(rulesets=[Ruleset("Bar", [Rule("bar test")])]),
            PromptTask(rulesets=[Ruleset("Baz", [Rule("baz test")])]),
        )

        assert isinstance(pipeline.tasks[0], PromptTask)
        assert len(pipeline.tasks[0].rulesets) == 2
        assert pipeline.tasks[0].rulesets[0].name == "Foo"
        assert pipeline.tasks[0].rulesets[1].name == "Bar"

        assert isinstance(pipeline.tasks[1], PromptTask)
        assert len(pipeline.tasks[1].rulesets) == 2
        assert pipeline.tasks[1].rulesets[0].name == "Foo"
        assert pipeline.tasks[1].rulesets[1].name == "Baz"

    def test_rules(self):
        pipeline = Pipeline(rules=[Rule("foo test")])

        pipeline.add_tasks(PromptTask(rules=[Rule("bar test")]), PromptTask(rules=[Rule("baz test")]))

        assert isinstance(pipeline.tasks[0], PromptTask)
        assert len(pipeline.tasks[0].rulesets) == 1
        assert pipeline.tasks[0].rulesets[0].name == "Default Ruleset"
        assert len(pipeline.tasks[0].rulesets[0].rules) == 2

        assert isinstance(pipeline.tasks[1], PromptTask)
        assert pipeline.tasks[1].rulesets[0].name == "Default Ruleset"
        assert len(pipeline.tasks[1].rulesets[0].rules) == 2

    def test_rules_and_rulesets(self):
        pipeline = Pipeline(rules=[Rule("foo test")], rulesets=[Ruleset("Bar", [Rule("bar test")])])
        assert len(pipeline.rulesets) == 2
        assert len(pipeline.rules) == 1

        pipeline = Pipeline()
        pipeline.add_task(PromptTask(rules=[Rule("foo test")], rulesets=[Ruleset("Bar", [Rule("bar test")])]))
        assert isinstance(pipeline.tasks[0], PromptTask)
        assert len(pipeline.tasks[0].rulesets) == 2
        assert len(pipeline.tasks[0].rules) == 1

    def test_with_no_task_memory(self):
        pipeline = Pipeline()

        pipeline.add_task(PromptTask(tools=[MockTool()]))

        assert isinstance(pipeline.tasks[0], PromptTask)
        assert pipeline.tasks[0].tools[0].input_memory is not None
        assert pipeline.tasks[0].tools[0].input_memory[0] == pipeline.task_memory
        assert pipeline.tasks[0].tools[0].output_memory is None

    def test_with_task_memory(self):
        pipeline = Pipeline()

        pipeline.add_task(PromptTask(tools=[MockTool(off_prompt=True)]))

        assert isinstance(pipeline.tasks[0], PromptTask)
        assert pipeline.tasks[0].task_memory == pipeline.task_memory
        assert pipeline.tasks[0].tools[0].input_memory is not None
        assert pipeline.tasks[0].tools[0].input_memory[0] == pipeline.task_memory
        assert pipeline.tasks[0].tools[0].output_memory is not None
        assert pipeline.tasks[0].tools[0].output_memory["test"][0] == pipeline.task_memory

    def test_with_task_memory_and_empty_tool_output_memory(self):
        pipeline = Pipeline()

        pipeline.add_task(PromptTask(tools=[MockTool(output_memory={}, off_prompt=True)]))

        assert isinstance(pipeline.tasks[0], PromptTask)
        assert pipeline.tasks[0].tools[0].output_memory == {}

    def test_without_task_memory(self):
        pipeline = Pipeline(task_memory=None)

        pipeline.add_task(PromptTask(tools=[MockTool()]))

        assert isinstance(pipeline.tasks[0], PromptTask)
        assert pipeline.tasks[0].tools[0].input_memory is None
        assert pipeline.tasks[0].tools[0].output_memory is None

    def test_with_memory(self):
        first_task = PromptTask("test1")
        second_task = PromptTask("test2")
        third_task = PromptTask("test3")

        pipeline = Pipeline(conversation_memory=ConversationMemory())

        pipeline + [first_task, second_task, third_task]

        assert pipeline.conversation_memory is not None
        assert len(pipeline.conversation_memory.runs) == 0

        pipeline.run()
        pipeline.run()
        pipeline.run()

        assert len(pipeline.conversation_memory.runs) == 3

    def test_tasks_initialization(self):
        first_task = PromptTask(id="test1")
        second_task = PromptTask(id="test2")
        third_task = PromptTask(id="test3")
        pipeline = Pipeline(tasks=[first_task, second_task, third_task])

        assert len(pipeline.tasks) == 3
        assert pipeline.tasks[0].id == "test1"
        assert pipeline.tasks[1].id == "test2"
        assert pipeline.tasks[2].id == "test3"
        assert len(first_task.parents) == 0
        assert len(first_task.children) == 1
        assert len(second_task.parents) == 1
        assert len(second_task.children) == 1
        assert len(third_task.parents) == 1
        assert len(third_task.children) == 0

    def test_tasks_order(self):
        first_task = PromptTask("test1")
        second_task = PromptTask("test2")
        third_task = PromptTask("test3")

        pipeline = Pipeline()

        pipeline + first_task
        pipeline + second_task
        pipeline + third_task

        assert pipeline.input_task.id is first_task.id
        assert pipeline.tasks[1].id is second_task.id
        assert pipeline.tasks[2].id is third_task.id
        assert pipeline.output_task.id is third_task.id

    def test_add_task(self):
        first_task = PromptTask("test1")
        second_task = PromptTask("test2")

        pipeline = Pipeline()

        pipeline + first_task
        pipeline + second_task

        assert len(pipeline.tasks) == 2
        assert first_task in pipeline.tasks
        assert second_task in pipeline.tasks
        assert first_task.structure == pipeline
        assert second_task.structure == pipeline
        assert len(first_task.parents) == 0
        assert len(first_task.children) == 1
        assert len(second_task.parents) == 1
        assert len(second_task.children) == 0

    def test_add_tasks(self):
        first_task = PromptTask("test1")
        second_task = PromptTask("test2")

        pipeline = Pipeline()

        pipeline + [first_task, second_task]

        assert len(pipeline.tasks) == 2
        assert first_task in pipeline.tasks
        assert second_task in pipeline.tasks
        assert first_task.structure == pipeline
        assert second_task.structure == pipeline
        assert len(first_task.parents) == 0
        assert len(first_task.children) == 1
        assert len(second_task.parents) == 1
        assert len(second_task.children) == 0

    def test_nested_tasks(self):
        pipeline = Pipeline(
            tasks=[
                [
                    PromptTask("parent", id=f"parent_{i}"),
                    PromptTask("child", id=f"child_{i}", parent_ids=[f"parent_{i}"]),
                    PromptTask("grandchild", id=f"grandchild_{i}", parent_ids=[f"child_{i}"]),
                ]
                for i in range(3)
            ]
        )

        pipeline.run()
        assert pipeline.output_task.id == "grandchild_2"
        assert len(pipeline.tasks) == 9

    def test_insert_task_in_middle(self):
        first_task = PromptTask("test1", id="test1")
        second_task = PromptTask("test2", id="test2")
        third_task = PromptTask("test3", id="test3")

        pipeline = Pipeline()

        pipeline + [first_task, second_task]
        pipeline.insert_task(first_task, third_task)

        assert len(pipeline.tasks) == 3
        assert first_task in pipeline.tasks
        assert second_task in pipeline.tasks
        assert third_task in pipeline.tasks
        assert first_task.structure == pipeline
        assert second_task.structure == pipeline
        assert third_task.structure == pipeline
        assert [parent.id for parent in first_task.parents] == []
        assert [child.id for child in first_task.children] == ["test3"]
        assert [parent.id for parent in second_task.parents] == ["test3"]
        assert [child.id for child in second_task.children] == []
        assert [parent.id for parent in third_task.parents] == ["test1"]
        assert [child.id for child in third_task.children] == ["test2"]

    def test_insert_task_at_end(self):
        first_task = PromptTask("test1", id="test1")
        second_task = PromptTask("test2", id="test2")
        third_task = PromptTask("test3", id="test3")

        pipeline = Pipeline()

        pipeline + [first_task, second_task]
        pipeline.insert_task(second_task, third_task)

        assert len(pipeline.tasks) == 3
        assert first_task in pipeline.tasks
        assert second_task in pipeline.tasks
        assert third_task in pipeline.tasks
        assert first_task.structure == pipeline
        assert second_task.structure == pipeline
        assert third_task.structure == pipeline
        assert [parent.id for parent in first_task.parents] == []
        assert [child.id for child in first_task.children] == ["test2"]
        assert [parent.id for parent in second_task.parents] == ["test1"]
        assert [child.id for child in second_task.children] == ["test3"]
        assert [parent.id for parent in third_task.parents] == ["test2"]
        assert [child.id for child in third_task.children] == []

    def test_prompt_stack_without_memory(self):
        pipeline = Pipeline(conversation_memory=None, rules=[Rule("test")])

        task1 = PromptTask("test")
        task2 = PromptTask("test")

        pipeline.add_tasks(task1, task2)

        assert len(task1.prompt_stack.messages) == 2
        assert len(task2.prompt_stack.messages) == 2

        pipeline.run()

        assert len(task1.prompt_stack.messages) == 3
        assert len(task2.prompt_stack.messages) == 3

        pipeline.run()

        assert len(task1.prompt_stack.messages) == 3
        assert len(task2.prompt_stack.messages) == 3

    def test_prompt_stack_with_memory(self):
        pipeline = Pipeline(rules=[Rule("test")])

        task1 = PromptTask("test")
        task2 = PromptTask("test")

        pipeline.add_tasks(task1, task2)

        assert len(task1.prompt_stack.messages) == 2
        assert len(task2.prompt_stack.messages) == 2

        pipeline.run()

        assert len(task1.prompt_stack.messages) == 5
        assert len(task2.prompt_stack.messages) == 5

        pipeline.run()

        assert len(task1.prompt_stack.messages) == 7
        assert len(task2.prompt_stack.messages) == 7

    def test_text_artifact_token_count(self):
        text = "foobar"

        assert TextArtifact(text).token_count(
            OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL)
        ) == OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL).count_tokens(text)

    def test_run(self):
        task = PromptTask("test")
        pipeline = Pipeline()
        pipeline + task

        assert task.state == BaseTask.State.PENDING

        result = pipeline.run()

        assert "mock output" in result.output_task.output.to_text()
        assert task.state == BaseTask.State.FINISHED

    def test_run_with_args(self):
        task = PromptTask("{{ args[0] }}-{{ args[1] }}")
        pipeline = Pipeline()
        pipeline + [task]

        pipeline._execution_args = ("test1", "test2")

        assert task.input.to_text() == "test1-test2"

        pipeline.run()

        assert task.input.to_text() == "-"

    def test_context(self):
        parent = PromptTask("parent")
        task = PromptTask("test")
        child = PromptTask("child")
        pipeline = Pipeline()

        pipeline + [parent, task, child]

        context = pipeline.context(task)

        assert context["parent_output"] is None

        pipeline.run("hello", foo="bar")

        context = pipeline.context(task)

        assert context["parent_output"] == parent.output
        assert context["task_outputs"] == pipeline.task_outputs
        assert context["structure"] == pipeline
        assert context["parent"] == parent
        assert context["child"] == child
        assert context["args"] == ("hello",)
        assert context["kwargs"] == {"foo": "bar"}

    def test_run_with_error_artifact(self, error_artifact_task, waiting_task):
        end_task = PromptTask("end")
        pipeline = Pipeline(tasks=[waiting_task, error_artifact_task, end_task])
        pipeline.run()

        with pytest.raises(ValueError, match="Structure's output Task has no output. Run"):
            assert pipeline.output

    def test_run_with_error_artifact_no_fail_fast(self, error_artifact_task, waiting_task):
        end_task = PromptTask("end")
        pipeline = Pipeline(tasks=[waiting_task, error_artifact_task, end_task], fail_fast=False)
        pipeline.run()

        assert pipeline.output is not None

    def test_add_duplicate_task(self):
        task = PromptTask("test")
        pipeline = Pipeline()

        pipeline + task
        pipeline + task

        assert len(pipeline.tasks) == 1

    def test_add_duplicate_task_directly(self):
        task = PromptTask("test")
        pipeline = Pipeline()

        pipeline + task
        pipeline._tasks.append(task)

        with pytest.raises(ValueError, match=f"Duplicate task with id {task.id} found."):
            pipeline.run()

    def test_task_outputs(self):
        task = PromptTask("test")
        pipeline = Pipeline()

        pipeline + [task]

        assert len(pipeline.task_outputs) == 1
        assert pipeline.task_outputs[task.id] is None
        pipeline.run()
        assert len(pipeline.task_outputs) == 1
        assert pipeline.task_outputs[task.id] == task.output
