import pytest
from pydantic import BaseModel
from schema import Literal, Optional, Schema

from tests.mocks.mock_tool.tool import MockTool
from tests.mocks.mock_tool_pydantic.tool import MockToolPydantic


class OutputSchema(BaseModel):
    foo: str


class TestActivityMixin:
    @pytest.fixture()
    def tool(self):
        return MockTool(test_field="hello", test_int=5)

    def test_activity_name(self, tool):
        assert tool.activity_name(tool.test) == "test"

    def test_activity_description(self, tool):
        description = tool.activity_description(tool.test)

        assert description == f"test description: {tool.foo()}"

    def test_activity_schema(self, tool):
        schema = tool.activity_schema(tool.test).json_schema("InputSchema")

        assert schema == Schema({"values": getattr(tool.test, "config")["schema"]}).json_schema("InputSchema")
        assert schema["properties"].get("artifact") is None

    def test_activity_with_no_schema(self, tool):
        assert tool.activity_schema(tool.test_no_schema) is None

    def test_find_activity(self):
        tool = MockTool(test_field="hello", test_int=5, allowlist=["test"])
        assert tool.find_activity("test") == tool.test
        assert tool.find_activity("test_str_output") is None

    def test_activities(self, tool):
        assert len(tool.activities()) == 8
        assert tool.activities()[0] == tool.test

    def test_allowlist_and_denylist_validation(self):
        with pytest.raises(ValueError):
            MockTool(test_field="hello", test_int=5, allowlist=[], denylist=[])

    def test_allowlist(self):
        tool = MockTool(test_field="hello", test_int=5, allowlist=["test"])

        assert len(tool.activities()) == 1

    def test_denylist(self):
        tool = MockTool(test_field="hello", test_int=5, denylist=["test"])

        assert len(tool.activities()) == 7

    def test_invalid_allowlist(self):
        with pytest.raises(ValueError):
            MockTool(test_field="hello", test_int=5, allowlist=["test_foo"])

    def test_invalid_denylist(self):
        with pytest.raises(ValueError):
            MockTool(test_field="hello", test_int=5, denylist=["test_foo"])

    def test_disable_activities(self, tool):
        assert len(tool.activities()) > 0

        tool.disable_activities()

        assert len(tool.activities()) == 0

    def test_enable_activities(self, tool):
        tool.disable_activities()

        assert len(tool.activities()) == 0

        tool.enable_activities()

        assert len(tool.activities()) > 0

    def test_extra_schema_properties(self):
        tool = MockTool(
            test_field="hello",
            test_int=5,
            extra_schema_properties={"test": {Literal("new_property"): str, Optional("optional_property"): int}},
        )
        schema = tool.activity_schema(tool.test).json_schema("InputSchema")

        assert schema == {
            "$id": "InputSchema",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {
                "values": {
                    "description": "Test input",
                    "properties": {
                        "test": {"type": "string"},
                        "new_property": {"type": "string"},
                        "optional_property": {"type": "integer"},
                    },
                    "required": ["test", "new_property"],
                    "additionalProperties": False,
                    "type": "object",
                }
            },
            "required": ["values"],
            "additionalProperties": False,
            "type": "object",
        }

    def test_callable_schema(self):
        tool = MockTool(custom_schema={"test": str})
        schema = tool.activity_schema(tool.test_callable_schema).json_schema("InputSchema")

        assert schema == {
            "$id": "InputSchema",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {
                "values": {
                    "description": "Test input",
                    "properties": {"test": {"type": "string"}},
                    "required": ["test"],
                    "additionalProperties": False,
                    "type": "object",
                }
            },
            "required": ["values"],
            "additionalProperties": False,
            "type": "object",
        }

    @pytest.mark.parametrize(
        ("tool", "expected_output"),
        [
            (
                MockTool(),
                {
                    "$id": "InputSchema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {
                        "values": {
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "additionalProperties": False,
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "additionalProperties": False,
                    "type": "object",
                },
            ),
            (
                MockToolPydantic(),
                {
                    "$id": "InputSchema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "properties": {
                                "foo": {
                                    "description": "Test input",
                                    "title": "Foo",
                                    "type": "string",
                                },
                            },
                            "required": ["foo"],
                            "title": "TestModel",
                            "type": "object",
                        },
                    },
                    "title": "TestModel",
                    "type": "object",
                },
            ),
        ],
    )
    @pytest.mark.parametrize("activity_name", ["test", "test_callable_schema"])
    def test_to_activity_json_schema(self, tool, activity_name, expected_output):
        json_schema = tool.to_activity_json_schema(getattr(tool, activity_name), "InputSchema")
        assert json_schema == expected_output
