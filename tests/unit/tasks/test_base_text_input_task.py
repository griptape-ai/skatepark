from griptape.artifacts import TextArtifact
from griptape.rules import Rule, Ruleset
from griptape.structures import Pipeline
from tests.mocks.mock_text_input_task import MockTextInputTask


class TestBaseTextInputTask:
    def test_string_input(self):
        assert MockTextInputTask("foobar").input.value == "foobar"

        task = MockTextInputTask()
        task.input = "foobar"
        assert task.input.value == "foobar"

    def test_artifact_input(self):
        assert MockTextInputTask(TextArtifact("foobar")).input.value == "foobar"

        task = MockTextInputTask()
        task.input = TextArtifact("foobar")
        assert task.input.value == "foobar"

    def test_callable_input(self):
        assert MockTextInputTask(lambda _: TextArtifact("foobar")).input.value == "foobar"

        task = MockTextInputTask()
        task.input = lambda _: TextArtifact("foobar")
        assert task.input.value == "foobar"

    def test_full_context(self):
        parent = MockTextInputTask("parent")
        subtask = MockTextInputTask("test", context={"foo": "bar"})
        child = MockTextInputTask("child")

        assert parent.full_context == {}
        assert subtask.full_context == {"foo": "bar"}
        assert child.full_context == {}

        pipeline = Pipeline()

        pipeline.add_tasks(parent, subtask, child)

        pipeline.run()

        context = subtask.full_context

        assert context["foo"] == "bar"
        assert context["parent_output"] == parent.output
        assert context["structure"] == pipeline
        assert context["parent"] == parent
        assert context["child"] == child

    def test_rulesets(self):
        prompt_task = MockTextInputTask(
            rulesets=[Ruleset("Foo", [Rule("foo test")]), Ruleset("Bar", [Rule("bar test")])]
        )

        assert len(prompt_task.all_rulesets) == 2
        assert prompt_task.all_rulesets[0].name == "Foo"
        assert prompt_task.all_rulesets[1].name == "Bar"

    def test_rules(self):
        prompt_task = MockTextInputTask(rules=[Rule("foo test"), Rule("bar test")])

        assert prompt_task.all_rulesets[0].name == "Default Ruleset"
