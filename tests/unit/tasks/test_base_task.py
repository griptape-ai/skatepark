from unittest.mock import Mock

import pytest

from griptape.artifacts import TextArtifact
from griptape.events import EventBus
from griptape.events.event_listener import EventListener
from griptape.structures import Agent, Workflow
from griptape.tasks import ActionsSubtask
from tests.mocks.mock_task import MockTask
from tests.mocks.mock_tool.tool import MockTool


class TestBaseTask:
    @pytest.fixture()
    def task(self):
        EventBus.add_event_listeners([EventListener(on_event=Mock())])
        agent = Agent(
            tools=[MockTool()],
        )
        EventBus.add_event_listeners([EventListener(on_event=Mock())])

        agent.add_task(MockTask("foobar", max_meta_memory_entries=2))

        return agent.task

    def test_meta_memories(self, task):
        subtask = ActionsSubtask()

        subtask.structure = task.structure

        assert len(task.meta_memories) == 0

        task.structure.task_memory.process_output(MockTool().test, subtask, TextArtifact("foo"))
        task.structure.task_memory.process_output(MockTool().test, subtask, TextArtifact("foo"))
        task.structure.task_memory.process_output(MockTool().test, subtask, TextArtifact("foo"))

        assert len(task.meta_memories) == 2

    def test_parent_outputs(self, task):
        parent_1 = MockTask("foobar1", id="foobar1")
        parent_2 = MockTask("foobar2", id="foobar2")
        parent_3 = MockTask("foobar3", id="foobar3")
        child = MockTask("foobar", id="foobar")

        child.add_parent(parent_1)
        child.add_parent(parent_2)
        child.add_parent(parent_3)

        workflow = Workflow(tasks=[parent_1, parent_2, parent_3, child])
        workflow.run()

        parent_3.output = None
        assert child.parent_outputs == {
            parent_1.id: parent_1.output,
            parent_2.id: parent_2.output,
        }

    def test_parents_output(self, task):
        parent_1 = MockTask("foobar1", id="foobar1")
        parent_2 = MockTask("foobar2", id="foobar2")
        parent_3 = MockTask("foobar3", id="foobar3")
        child = MockTask("foobar", id="foobar")

        child.add_parent(parent_1)
        child.add_parent(parent_2)
        child.add_parent(parent_3)

        workflow = Workflow(tasks=[parent_1, parent_2, parent_3, child])
        workflow.run()

        parent_2.output = None

        assert child.parents_output_text == "foobar1\nfoobar3"

    def test_parents_property_no_structure(self, task):
        workflow = Workflow()
        task1 = MockTask("foobar1", id="foobar1")
        task2 = MockTask("foobar2", id="foobar2")
        task3 = MockTask("foobar3", id="foobar3")
        child = MockTask("foobar", id="foobar")

        child.add_parent(task1)
        child.add_parent(task2)
        child.add_parent(task3)

        with pytest.raises(ValueError, match="Structure must be set to access parents"):
            child.parents  # noqa: B018

        workflow.add_tasks(task1, task2, task3, child)
        child.structure = workflow

        assert len(child.parents) == 3

    def test_children_property_no_structure(self, task):
        workflow = Workflow()
        task1 = MockTask("foobar1", id="foobar1")
        task2 = MockTask("foobar2", id="foobar2")
        task3 = MockTask("foobar3", id="foobar3")
        parent = MockTask("foobar", id="foobar")

        parent.add_child(task1)
        parent.add_child(task2)
        parent.add_child(task3)

        with pytest.raises(ValueError, match="Structure must be set to access children"):
            parent.children  # noqa: B018

        workflow.add_tasks(task1, task2, task3, parent)
        parent.structure = workflow

        assert len(parent.children) == 3

    def test_run_publish_events(self, task):
        task.run()

        assert EventBus.event_listeners[0].on_event.call_count == 2

    def test_add_parent(self, task):
        agent = Agent()
        parent = MockTask("parent foobar", id="parent_foobar", structure=agent)

        result = task.add_parent(parent)
        result = task.add_parent(parent)

        assert parent.id in task.parent_ids
        assert task.id in parent.child_ids
        assert result == task

        assert agent.tasks == [parent]

    def test_add_child(self, task):
        agent = Agent()
        child = MockTask("child foobar", id="child_foobar", structure=agent)

        result = task.add_child(child)
        result = task.add_child(child)

        assert child.id in task.child_ids
        assert task.id in child.parent_ids
        assert result == task

        assert agent.tasks == [child]

    def test_add_parent_bitshift(self, task):
        parent = MockTask("parent foobar", id="parent_foobar")

        added_task = task << parent

        assert parent.id in task.parent_ids
        assert task.id in parent.child_ids
        assert added_task == parent

    def test_add_multiple_parent_bitshift(self, task):
        parent_1 = MockTask("parent foobar", id="parent_1_foobar")
        parent_2 = MockTask("parent foobar", id="parent_2_foobar")

        added_tasks = task << [parent_1, parent_2]

        assert parent_1.id in task.parent_ids
        assert parent_2.id in task.parent_ids
        assert task.id in parent_1.child_ids
        assert task.id in parent_2.child_ids
        assert added_tasks == [parent_1, parent_2]

    def test_add_child_bitshift(self, task):
        child = MockTask("child foobar", id="child_foobar")

        added_task = task >> child

        assert child.id in task.child_ids
        assert task.id in child.parent_ids
        assert added_task == child

    def test_add_multiple_child_bitshift(self, task):
        child_1 = MockTask("child foobar", id="child_1_foobar")
        child_2 = MockTask("child foobar", id="child_2_foobar")

        added_tasks = task >> [child_1, child_2]

        assert child_1.id in task.child_ids
        assert child_2.id in task.child_ids
        assert task.id in child_1.parent_ids
        assert task.id in child_2.parent_ids
        assert added_tasks == [child_1, child_2]

    def test_to_dict(self, task):
        expected_task_dict = {
            "type": task.type,
            "id": task.id,
            "state": str(task.state),
            "parent_ids": task.parent_ids,
            "child_ids": task.child_ids,
            "max_meta_memory_entries": task.max_meta_memory_entries,
            "context": task.context,
        }
        assert expected_task_dict == task.to_dict()

    def test_from_dict(self):
        task = MockTask("Foobar2", id="Foobar2")

        serialized_task = task.to_dict()
        assert isinstance(serialized_task, dict)

        deserialized_task = MockTask.from_dict(serialized_task)
        assert isinstance(deserialized_task, MockTask)

        workflow = Workflow()
        workflow.add_task(deserialized_task)

        assert workflow.tasks == [deserialized_task]

        workflow.run()

        assert str(workflow.tasks[0].state) == "State.FINISHED"
        assert workflow.tasks[0].id == deserialized_task.id
        assert workflow.tasks[0].output.value == "foobar"

    def test_runnable_mixin(self):
        mock_on_before_run = Mock()
        mock_after_run = Mock()
        task = MockTask("foobar", on_before_run=mock_on_before_run, on_after_run=mock_after_run)

        task.run()

        mock_on_before_run.assert_called_once_with(task)
        mock_after_run.assert_called_once_with(task)

    def test_full_context(self, task):
        task.structure = Agent()
        task.structure._execution_args = ("foo", "bar")
        task.structure._execution_kwargs = {"baz": "qux"}

        assert task.full_context == {"args": ("foo", "bar"), "kwargs": {"baz": "qux"}, "structure": task.structure}
        assert task.structure.execution_args == ("foo", "bar")
        assert task.structure.execution_kwargs == {"baz": "qux"}

        task.structure = None
        task._execution_args = ("foo", "bar")
        task._execution_kwargs = {"baz": "qux"}

        assert task.full_context == {"args": ("foo", "bar"), "kwargs": {"baz": "qux"}}
        assert task.execution_args == ("foo", "bar")
        assert task.execution_kwargs == {"baz": "qux"}

    def test_is_pending(self, task):
        task.state = task.State.PENDING
        assert task.is_pending()

    def test_is_running(self, task):
        task.state = task.State.RUNNING
        assert task.is_running()

    def test_is_finished(self, task):
        task.state = task.State.FINISHED
        assert task.is_finished()

    def test___str__(self, task):
        task.output = TextArtifact("foobar")
        assert str(task) == "foobar"
        task.output = None
        assert str(task) == ""

    def test_run_args(self, task):
        task.run("foo", "bar")

        assert task._execution_args == ("foo", "bar")

    def test_run_kwargs(self, task):
        task.run(foo="bar")

        assert task._execution_kwargs == {"foo": "bar"}

    def test_args_full_context(self):
        task = MockTask()
        task.context = {"foo": "buzz"}
        task.run("foo", "bar", baz="qux")

        assert task.full_context["args"] == ("foo", "bar")
        assert task.full_context["kwargs"] == {"baz": "qux"}
