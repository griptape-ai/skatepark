from __future__ import annotations
import concurrent.futures as futures
from graphlib import TopologicalSorter
from typing import Any
from attr import define, field, Factory
from griptape.artifacts import ErrorArtifact
from griptape.structures import Structure
from griptape.tasks import BaseTask
from griptape.memory.structure import Run


@define
class Workflow(Structure):
    futures_executor: futures.Executor = field(default=Factory(lambda: futures.ThreadPoolExecutor()), kw_only=True)

    def add_task(self, task: BaseTask) -> BaseTask:
        task.preprocess(self)

        if self.output_task:
            self.output_task.child_ids.append(task.id)
            task.parent_ids.append(self.output_task.id)

        self.tasks.append(task)

        return task

    def insert_task(
        self,
        parent_task: BaseTask,
        task: BaseTask,
        child_task: BaseTask,
        preserve_relationship: bool = False,
    ) -> BaseTask:
        return self.insert_tasks(
            parent_task, [task], child_task, preserve_relationship
        )[0]

    def insert_tasks(
        self,
        parent_task: BaseTask,
        tasks: list[BaseTask],
        child_task: BaseTask,
        preserve_relationship: bool = False,
    ) -> list[BaseTask]:
        """Insert a task between two tasks in the workflow.

        Args:
            parent_task: The task that will be the parent of the new task.
            child_task: The task that will be the child of the new task.
            task: The task to insert.
            preserve_relationship: Whether to preserve the parent/child relationship when inserting between parent and child tasks.
        """
        for task in tasks:
            task.preprocess(self)

            if parent_task.id not in task.parent_ids:
                task.parent_ids.append(parent_task.id)
            if child_task.id not in task.child_ids:
                task.child_ids.append(child_task.id)

            if task.id not in parent_task.child_ids:
                parent_task.child_ids.append(task.id)
            if task.id not in child_task.parent_ids:
                child_task.parent_ids.append(task.id)

            if not preserve_relationship:
                if child_task.id in parent_task.child_ids:
                    parent_task.child_ids.remove(child_task.id)
                if parent_task.id in child_task.parent_ids:
                    child_task.parent_ids.remove(parent_task.id)

            parent_index = self.tasks.index(parent_task)

            self.tasks.insert(parent_index + 1, task)

        return tasks

    def try_run(self, *args) -> Workflow:
        self._execution_args = args
        ordered_tasks = self.order_tasks()
        exit_loop = False

        while not self.is_finished() and not exit_loop:
            futures_list = {}

            for task in ordered_tasks:
                if task.can_execute():
                    future = self.futures_executor.submit(task.execute)
                    futures_list[future] = task

            # Wait for all tasks to complete
            for future in futures.as_completed(futures_list):
                if isinstance(future.result(), ErrorArtifact):
                    exit_loop = True

                    break

        if self.memory:
            run = Run(input=self.input_task.input.to_text(), output=self.output_task.output.to_text())

            self.memory.add_run(run)

        self._execution_args = ()

        return self

    def context(self, task: BaseTask) -> dict[str, Any]:
        context = super().context(task)

        context.update(
            {
                "parent_outputs": {
                    parent.id: parent.output.to_text() if parent.output else "" for parent in task.parents
                },
                "parents": {parent.id: parent for parent in task.parents},
                "children": {child.id: child for child in task.children},
            }
        )

        return context

    def to_graph(self) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}

        for key_task in self.tasks:
            graph[key_task.id] = set()

            for value_task in self.tasks:
                if key_task.id in value_task.child_ids:
                    graph[key_task.id].add(value_task.id)

        return graph

    def order_tasks(self) -> list[BaseTask]:
        return [self.find_task(task_id) for task_id in TopologicalSorter(self.to_graph()).static_order()]
