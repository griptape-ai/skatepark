from __future__ import annotations

from attrs import define, field
from typing import TYPE_CHECKING

from griptape.artifacts import BaseArtifact
from griptape.mixins import SerializableMixin

if TYPE_CHECKING:
    from griptape.common import Action


@define()
class ActionArtifact(BaseArtifact, SerializableMixin):
    value: Action = field(metadata={"serializable": True})

    def __add__(self, other: BaseArtifact) -> ActionArtifact:
        raise NotImplementedError
