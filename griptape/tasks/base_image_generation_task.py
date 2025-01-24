from __future__ import annotations

import logging
import os
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import Factory, define, field

from griptape.configs import Defaults
from griptape.loaders import ImageLoader
from griptape.mixins.artifact_file_output_mixin import ArtifactFileOutputMixin
from griptape.mixins.rule_mixin import RuleMixin
from griptape.rules import Rule, Ruleset
from griptape.tasks import BaseTask

if TYPE_CHECKING:
    from griptape.artifacts import ImageArtifact

logger = logging.getLogger(Defaults.logging_config.logger_name)


if TYPE_CHECKING:
    from griptape.drivers.image_generation import BaseImageGenerationDriver


@define
class BaseImageGenerationTask(ArtifactFileOutputMixin, RuleMixin, BaseTask, ABC):
    """Provides a base class for image generation-related tasks.

    Attributes:
        negative_rulesets: List of negatively-weighted rulesets applied to the text prompt, if supported by the driver.
        negative_rules: List of negatively-weighted rules applied to the text prompt, if supported by the driver.
        output_dir: If provided, the generated image will be written to disk in output_dir.
        output_file: If provided, the generated image will be written to disk as output_file.
    """

    DEFAULT_NEGATIVE_RULESET_NAME = "Negative Ruleset"

    image_generation_driver: BaseImageGenerationDriver = field(
        default=Factory(lambda: Defaults.drivers_config.image_generation_driver),
        kw_only=True,
    )
    _negative_rulesets: list[Ruleset] = field(factory=list, kw_only=True, alias="negative_rulesets")
    negative_rules: list[Rule] = field(factory=list, kw_only=True)

    @property
    def negative_rulesets(self) -> list[Ruleset]:
        negative_rulesets = self._negative_rulesets

        if self.negative_rules:
            negative_rulesets.append(Ruleset(name=self.DEFAULT_NEGATIVE_RULESET_NAME, rules=self.negative_rules))

        return negative_rulesets

    def _read_from_file(self, path: str) -> ImageArtifact:
        logger.info("Reading image from %s", os.path.abspath(path))
        return ImageLoader().load(Path(path))

    def _get_prompts(self, prompt: str) -> list[str]:
        return [prompt, *[rule.value for ruleset in self.all_rulesets for rule in ruleset.rules]]

    def _get_negative_prompts(self) -> list[str]:
        return [rule.value for ruleset in self.negative_rulesets for rule in ruleset.rules]
