import json
from typing import Optional, Callable
import stringcase
import yaml
from attr import define, field
from fastapi import FastAPI
from griptape.api.extensions import BaseApiExtension
from griptape.artifacts import ErrorArtifact
from griptape.tools import BaseTool


@define
class ApiGenerator:
    host: str = field()
    tool: BaseTool = field(kw_only=True)
    path_prefix: Optional[str] = field(default=None, kw_only=True)
    extensions: list[BaseApiExtension] = field(factory=list, kw_only=True)
    api: FastAPI = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.api = self.generate_api()

    @property
    def full_host_path(self) -> str:
        return "/".join(
            s.strip("/") for s in [self.host, self.path_prefix] if s
        )

    def generate_yaml_api_spec(self) -> str:
        return yaml.safe_dump(self.api.openapi())

    def generate_json_api_spec(self) -> str:
        return json.dumps(self.api.openapi())

    def generate_api(self) -> FastAPI:
        api = FastAPI()
        api.title = f"{self.tool.name} API"

        for activity in self.tool.activities():
            api.add_api_route(
                path=stringcase.spinalcase(self.tool.activity_name(activity)),
                endpoint=self.execute_activity_fn(activity),
                methods=["GET"],
                operation_id=stringcase.pascalcase(self.tool.activity_name(activity)),
                description=self.tool.activity_description(activity)
            )

        for extension in self.extensions:
            extension.extend(self)

        return api

    def execute_activity_fn(self, action: Callable) -> Callable:
        def execute_activity(value: str) -> dict:
            try:
                return action(value).to_dict()
            except Exception as e:
                return ErrorArtifact(str(e)).to_dict()

        return execute_activity
