from __future__ import annotations
from griptape.artifacts import BaseArtifact
from typing import Any
import json
from attrs import define, field


@define
class GenericArtifact(BaseArtifact):
    value: Any = field()

    @classmethod
    def value_to_bytes(cls, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        else:
            return str(value).encode()

    @classmethod
    def value_to_dict(cls, value: Any) -> dict:
        dict_value = value if isinstance(value, dict) else json.loads(value)

        return {k: v for k, v in dict_value.items()}

    def to_text(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return self.to_text()

    def __bool__(self) -> bool:
        return bool(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __add__(self, other: BaseArtifact) -> BaseArtifact:
        raise NotImplementedError
