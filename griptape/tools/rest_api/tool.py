from __future__ import annotations

from textwrap import dedent
from typing import Optional
from urllib.parse import urljoin

from attrs import define, field
from schema import Schema

from griptape.artifacts import BaseArtifact, ErrorArtifact, JsonArtifact
from griptape.tools import BaseTool
from griptape.utils.decorators import activity


@define
class RestApiTool(BaseTool):
    """A tool for making REST API requests.

    Attributes:
        base_url: The base url that will be used for the request.
        path: The resource path that will be appended to base_url.
        description: A description of what the REST API does.
        request_body_schema: A JSON schema string describing the request body. Recommended for PUT, POST, and PATCH requests.
        request_query_params_schema: A JSON schema string describing the available query parameters.
        request_path_params_schema: A JSON schema string describing the available path parameters. The schema must describe an array of string values.
        request_headers: Headers to include in the requests.
    """

    QUERY_PARAMS_KEY = "query_params"
    PATH_PARAMS_KEY = "path_params"
    BODY_KEY = "body"

    base_url: str = field(kw_only=True)
    path: Optional[str] = field(default=None, kw_only=True)
    description: str = field(kw_only=True)
    timeout: int = field(default=30, kw_only=True)
    request_query_params_schema: Optional[dict] = field(default=None, kw_only=True)
    request_body_schema: Optional[dict] = field(default=None, kw_only=True)
    request_headers: Optional[dict[str, str]] = field(default=None, kw_only=True)

    @activity(
        config={"description": "{{ _self._build_description('put') }}", "schema": Schema({})},
    )
    def put(self, params: dict) -> BaseArtifact:
        from requests import exceptions, put

        values = params["values"]
        args = self._build_args(values)
        url = self._build_url(self.base_url, path=self.path)

        try:
            response = put(url, **args, timeout=self.timeout, headers=self.request_headers)

            return JsonArtifact(response.json(), meta={"status_code": response.status_code})
        except exceptions.RequestException as err:
            return ErrorArtifact(str(err))

    @activity(
        config={
            "description": "{{ _self._build_description('patch') }}",
            "schema": Schema({}),
        },
    )
    def patch(self, params: dict) -> BaseArtifact:
        from requests import exceptions, patch

        values = params["values"]
        path_params = values.get("path_params", [])
        args = self._build_args(values)
        url = self._build_url(self.base_url, path=self.path, path_params=path_params)

        try:
            response = patch(url, **args, timeout=self.timeout, headers=self.request_headers)
            return JsonArtifact(response.json(), meta={"status_code": response.status_code})
        except exceptions.RequestException as err:
            return ErrorArtifact(str(err))

    @activity(
        config={"description": "{{ _self._build_description('post') }}", "schema": Schema({})},
    )
    def post(self, params: dict) -> BaseArtifact:
        from requests import exceptions, post

        values = params["values"]
        args = self._build_args(values)
        url = self._build_url(self.base_url, path=self.path)

        try:
            response = post(url, **args, timeout=self.timeout, headers=self.request_headers)
            return JsonArtifact(response.json(), meta={"status_code": response.status_code})
        except exceptions.RequestException as err:
            return ErrorArtifact(str(err))

    @activity(
        config={
            "description": "{{ _self._build_description('get') }}",
            "schema": Schema({}),
        },
    )
    def get(self, params: dict) -> BaseArtifact:
        from requests import exceptions, get

        values = params["values"]

        path_params = values.get("path_params", [])
        args = self._build_args(values)
        url = self._build_url(self.base_url, path=self.path, path_params=path_params)

        try:
            response = get(url, **args, timeout=self.timeout, headers=self.request_headers)
            return JsonArtifact(response.json(), meta={"status_code": response.status_code})
        except exceptions.RequestException as err:
            return ErrorArtifact(str(err))

    @activity(
        config={"description": "{{ _self._build_description('delete') }}", "schema": Schema({})},
    )
    def delete(self, params: dict) -> BaseArtifact:
        from requests import delete, exceptions

        values = params["values"]

        path_params = values.get("path_params", [])
        args = self._build_args(values)
        url = self._build_url(self.base_url, path=self.path, path_params=path_params)

        try:
            response = delete(url, **args, timeout=self.timeout, headers=self.request_headers)
            return JsonArtifact(response.json(), meta={"status_code": response.status_code})
        except exceptions.RequestException as err:
            return ErrorArtifact(str(err))

    def _build_url(self, base_url: str, path: Optional[str] = None, path_params: Optional[list] = None) -> str:
        url = ""

        if path:
            url += path.strip("/")

        if path_params:
            url += f'/{str.join("/", map(str, path_params))}'

        return urljoin(base_url.strip("/"), url)

    def _build_args(self, values: dict) -> dict:
        query_params = values.get("query_params", {})
        body = values.get("body", {})

        return {"params": query_params, "json": body}

    def _build_description(self, method: str) -> str:
        full_url = self._build_url(self.base_url, path=self.path)
        return dedent(f"""
                This tool can be used to make a {method} request to the rest api url: { full_url }
                This rest api has the following description: { self.description }""")
