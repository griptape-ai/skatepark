import pytest
import schema
from pydantic import BaseModel, create_model

from griptape.artifacts.image_artifact import ImageArtifact
from griptape.artifacts.list_artifact import ListArtifact
from griptape.artifacts.text_artifact import TextArtifact
from griptape.memory.structure import ConversationMemory
from griptape.memory.structure.run import Run
from griptape.rules import Rule
from griptape.rules.json_schema_rule import JsonSchemaRule
from griptape.rules.ruleset import Ruleset
from griptape.structures import Pipeline
from griptape.tasks import PromptTask
from tests.mocks.mock_prompt_driver import MockPromptDriver
from tests.mocks.mock_tool.tool import MockTool


class TestPromptTask:
    def test_run(self):
        task = PromptTask("test")
        pipeline = Pipeline()

        pipeline.add_task(task)

        assert task.run().to_text() == "mock output"

    def test_to_text(self):
        task = PromptTask("{{ test }}", context={"test": "test value"})

        Pipeline().add_task(task)

        assert task.input.to_text() == "test value"

    def test_config_prompt_driver(self):
        task = PromptTask("test")
        Pipeline().add_task(task)

        assert isinstance(task.prompt_driver, MockPromptDriver)

    def test_input(self):
        # Structure context
        pipeline = Pipeline()
        task = PromptTask()
        pipeline.add_task(task)
        pipeline._execution_args = ("foo", "bar")
        assert task.input.value == "foo"
        pipeline._execution_args = ("fizz", "buzz")
        assert task.input.value == "fizz"

        # Str
        task = PromptTask("test")

        assert task.input.value == "test"

        # List of strs
        task = PromptTask(["test1", "test2"])

        assert task.input.value[0].value == "test1"
        assert task.input.value[1].value == "test2"

        # Tuple of strs
        task = PromptTask(("test1", "test2"))

        assert task.input.value[0].value == "test1"
        assert task.input.value[1].value == "test2"

        # Image artifact
        task = PromptTask(ImageArtifact(b"image-data", format="png", width=100, height=100))

        assert isinstance(task.input, ImageArtifact)
        assert task.input.value == b"image-data"
        assert task.input.format == "png"
        assert task.input.width == 100
        assert task.input.height == 100

        # List of str and image artifact
        task = PromptTask(["foo", ImageArtifact(b"image-data", format="png", width=100, height=100)])

        assert isinstance(task.input, ListArtifact)
        assert task.input.value[0].value == "foo"
        assert isinstance(task.input.value[1], ImageArtifact)
        assert task.input.value[1].value == b"image-data"
        assert task.input.value[1].format == "png"
        assert task.input.value[1].width == 100

        # List of str and nested image artifact
        task = PromptTask(["foo", [ImageArtifact(b"image-data", format="png", width=100, height=100)]])
        assert isinstance(task.input, ListArtifact)
        assert task.input.value[0].value == "foo"
        assert isinstance(task.input.value[1], ListArtifact)
        assert isinstance(task.input.value[1].value[0], ImageArtifact)
        assert task.input.value[1].value[0].value == b"image-data"
        assert task.input.value[1].value[0].format == "png"
        assert task.input.value[1].value[0].width == 100

        # Tuple of str and image artifact
        task = PromptTask(("foo", ImageArtifact(b"image-data", format="png", width=100, height=100)))

        assert isinstance(task.input, ListArtifact)
        assert task.input.value[0].value == "foo"
        assert isinstance(task.input.value[1], ImageArtifact)
        assert task.input.value[1].value == b"image-data"
        assert task.input.value[1].format == "png"
        assert task.input.value[1].width == 100

        # Lambda returning list of str and image artifact
        task = PromptTask(
            ListArtifact([TextArtifact("foo"), ImageArtifact(b"image-data", format="png", width=100, height=100)])
        )

        assert isinstance(task.input, ListArtifact)
        assert task.input.value[0].value == "foo"
        assert isinstance(task.input.value[1], ImageArtifact)
        assert task.input.value[1].value == b"image-data"
        assert task.input.value[1].format == "png"
        assert task.input.value[1].width == 100

        # Lambda returning list of str and image artifact
        task = PromptTask(
            lambda _: ListArtifact(
                [TextArtifact("foo"), ImageArtifact(b"image-data", format="png", width=100, height=100)]
            )
        )
        assert isinstance(task.input, ListArtifact)
        assert task.input.value[0].value == "foo"
        assert isinstance(task.input.value[1], ImageArtifact)
        assert task.input.value[1].value == b"image-data"
        assert task.input.value[1].format == "png"
        assert task.input.value[1].width == 100

        # default case
        task = PromptTask({"default": "test"})

        assert task.input.value == str({"default": "test"})

    def test_input_context(self):
        pipeline = Pipeline(
            tasks=[
                PromptTask(
                    "foo",
                    prompt_driver=MockPromptDriver(),
                    on_before_run=lambda task: task.children[0].input,
                ),
                PromptTask("{{ parent_output }}", prompt_driver=MockPromptDriver()),
            ]
        )

        pipeline.run()

        assert pipeline.tasks[1].input.value == "mock output"

    def test_prompt_stack(self):
        task = PromptTask("{{ test }}", context={"test": "test value"}, rules=[Rule("test rule")])

        Pipeline().add_task(task)

        assert len(task.prompt_stack.messages) == 2
        assert task.prompt_stack.messages[0].is_system()
        assert task.prompt_stack.messages[1].is_user()

    def test_prompt_stack_empty_system_content(self):
        task = PromptTask("{{ test }}", context={"test": "test value"})

        pipeline = Pipeline(
            conversation_memory=ConversationMemory(
                runs=[Run(input=TextArtifact("input"), output=TextArtifact("output"))]
            )
        )
        pipeline.add_task(task)

        assert len(task.prompt_stack.messages) == 3
        assert task.prompt_stack.messages[0].is_user()
        assert task.prompt_stack.messages[0].to_text() == "input"
        assert task.prompt_stack.messages[1].is_assistant()
        assert task.prompt_stack.messages[1].to_text() == "output"
        assert task.prompt_stack.messages[2].is_user()
        assert task.prompt_stack.messages[2].to_text() == "test value"

    def test_prompt_stack_empty_native_schema(self):
        task = PromptTask(
            input="foo",
            prompt_driver=MockPromptDriver(),
            rules=[JsonSchemaRule({"foo": {}})],
        )

        assert task.prompt_stack.output_schema is None

    def test_rulesets(self):
        pipeline = Pipeline(
            rulesets=[Ruleset("Pipeline Ruleset")],
            rules=[Rule("Pipeline Rule")],
        )
        task = PromptTask(rulesets=[Ruleset("Task Ruleset")], rules=[Rule("Task Rule")])

        pipeline.add_task(task)

        assert len(task.rulesets) == 3
        assert task.rulesets[0].name == "Pipeline Ruleset"
        assert task.rulesets[1].name == "Task Ruleset"
        assert task.rulesets[2].name == "Default Ruleset"

        assert len(task.rulesets[0].rules) == 0
        assert len(task.rulesets[1].rules) == 0
        assert task.rulesets[2].rules[0].value == "Pipeline Rule"
        assert task.rulesets[2].rules[1].value == "Task Rule"

    def test_conversation_memory(self):
        conversation_memory = ConversationMemory()
        task = PromptTask("{{ test }}", context={"test": "test value"})

        task.run()
        task.run()

        assert len(conversation_memory.runs) == 0
        assert len(task.prompt_stack.messages) == 2

        task.conversation_memory = conversation_memory

        task.run()
        task.run()

        assert len(conversation_memory.runs) == 2
        assert len(task.prompt_stack.messages) == 6

        task.conversation_memory = None

        task.run()
        task.run()

        assert len(conversation_memory.runs) == 2
        assert len(task.prompt_stack.messages) == 2

    def test_subtasks(self):
        task = PromptTask(
            input="foo",
            prompt_driver=MockPromptDriver(),
        )

        task.run()
        assert len(task.subtasks) == 0

        task = PromptTask(input="foo", prompt_driver=MockPromptDriver(use_native_tools=True), tools=[MockTool()])

        task.run()
        assert len(task.subtasks) == 2

    @pytest.mark.parametrize("structured_output_strategy", ["native"])
    @pytest.mark.parametrize(
        ("output_schema", "expected_output"),
        [
            (schema.Schema({"foo": str}), {"foo": "bar"}),
            (create_model("Test", foo=(str, ...)), create_model("Test", foo=(str, ...))(foo="bar")),
            (None, "mock output"),
            ("foo", "Unsupported output schema type: <class 'str'>"),
        ],
    )
    def test_parse_output_schema(self, structured_output_strategy, output_schema, expected_output):
        task = PromptTask(
            input="foo",
            prompt_driver=MockPromptDriver(
                structured_output_strategy=structured_output_strategy,
                mock_structured_output={"foo": "bar"},
                max_attempts=0,
            ),
            output_schema=output_schema,
        )

        output = task.run()

        if isinstance(output.value, BaseModel) and isinstance(expected_output, BaseModel):
            assert output.value.model_dump_json() == expected_output.model_dump_json()
        else:
            assert output.value == expected_output
