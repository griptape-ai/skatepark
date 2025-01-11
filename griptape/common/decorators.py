from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, cast

import wrapt

from griptape.artifacts.base_artifact import BaseArtifact

T = TypeVar("T")


def observable(*dargs: Any, **dkwargs: Any) -> Callable:
    @wrapt.decorator
    def decorator(wrapped: Callable[..., T], instance: Any, args: Any, kwargs: Any) -> T:
        from griptape.common.observable import Observable
        from griptape.observability.observability import Observability

        return cast(
            T,
            Observability.observe(
                Observable.Call(
                    func=wrapped,
                    instance=instance,
                    args=args,
                    kwargs=kwargs,
                    decorator_args=dargs,
                    decorator_kwargs=dkwargs,
                )
            ),
        )

    # Check if it's being called as @observable or @observable(...)
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:  # pyright: ignore[reportArgumentType]
        # Case when decorator is used without arguments
        func = dargs[0]
        dargs = ()
        dkwargs = {}
        return decorator(func)  # pyright: ignore[reportCallIssue]
    else:
        # Case when decorator is used with arguments
        return decorator


def task(**kwargs: Any) -> Any:
    def decorator(func: Callable) -> Any:
        from griptape.artifacts import GenericArtifact

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            from griptape.tasks import BaseTask, CodeExecutionTask

            def on_run(task: Any) -> Any:
                arg_values = [arg.output.value for arg in args if isinstance(arg, BaseTask) if arg.output]
                output = func(*arg_values)

                if isinstance(output, BaseArtifact):
                    return output
                else:
                    return GenericArtifact(output)

            task = CodeExecutionTask(id=func.__name__, on_run=on_run)

            for arg in args:
                if isinstance(arg, BaseTask):
                    task.add_parent(arg)

            return task

        return wrapper

    return decorator
