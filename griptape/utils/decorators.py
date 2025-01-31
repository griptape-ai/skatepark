from __future__ import annotations

import functools
import inspect
from collections import OrderedDict
from typing import Any, Callable, Optional, cast

import schema
from pydantic import BaseModel
from schema import Schema

CONFIG_SCHEMA = Schema(
    {
        schema.Optional("name"): str,
        "description": str,
        schema.Optional("schema"): lambda data: isinstance(data, (Schema, Callable))
        or (isinstance(data, type) and issubclass(data, BaseModel)),
    }
)


def activity(config: dict) -> Any:
    validated_config = CONFIG_SCHEMA.validate(config)

    validated_config.update({k: v for k, v in config.items() if k not in validated_config})

    if not validated_config.get("schema"):
        validated_config["schema"] = None

    def decorator(func: Callable) -> Any:
        @functools.wraps(func)
        def wrapper(self: Any, params: dict) -> Any:
            return func(self, **_build_kwargs(func, params))

        setattr(wrapper, "name", validated_config.get("name", func.__name__))
        setattr(wrapper, "config", validated_config)
        setattr(wrapper, "is_activity", True)

        return wrapper

    return decorator


def lazy_property(attr_name: Optional[str] = None) -> Callable[[Callable[[Any], Any]], property]:
    def decorator(func: Callable[[Any], Any]) -> property:
        actual_attr_name = f"_{func.__name__}" if attr_name is None else attr_name

        @property
        @functools.wraps(func)
        def lazy_attr(self: Any) -> Any:
            if getattr(self, actual_attr_name) is None:
                setattr(self, actual_attr_name, func(self))
            return getattr(self, actual_attr_name)

        @lazy_attr.setter
        def lazy_attr(self: Any, value: Any) -> None:
            setattr(self, actual_attr_name, value)

        return lazy_attr

    return decorator


def _build_kwargs(func: Callable, params: dict) -> dict:
    func_params = cast(OrderedDict, inspect.signature(func).parameters.copy())
    func_params.popitem(last=False)

    kwarg_var = None
    for param in func_params.values():
        # if there is a **kwargs parameter, we can safely
        # pass all the params to the function
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            kwarg_var = func_params.pop(param.name).name
            break

    # only pass the values that are in the function signature
    # or if there is a **kwargs parameter, pass all the values
    kwargs = {k: v for k, v in params.get("values", {}).items() if k in func_params or kwarg_var is not None}

    # add 'params' and 'values' if they are in the signature
    # or if there is a **kwargs parameter
    if "params" in func_params or kwarg_var is not None:
        kwargs["params"] = params
    if "values" in func_params or kwarg_var is not None:
        kwargs["values"] = params.get("values")

    # set any missing parameters to None
    for param_name, param in func_params.items():
        if param_name not in kwargs and param.default == inspect.Parameter.empty:
            kwargs[param_name] = None

    return kwargs
