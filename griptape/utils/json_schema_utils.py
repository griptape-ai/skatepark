from __future__ import annotations

from copy import deepcopy
from typing import Any


def build_strict_schema(json_schema: dict, schema_id: str) -> dict:
    """Performs a series of post-processing steps to ensure a JSON schema is compatible with LLMs.

    1. Adds the `$id` and `$schema` keys.
    2. Sets `additionalProperties` to `False` for objects without this key.
    3. Resolves `$ref`s and removes `$defs`.

    Args:
        json_schema: The JSON schema to ensure is strict.
        schema_id: The ID of the schema.

    Returns: The strict JSON schema.
    """
    from griptape.utils.dict_utils import add_key_in_dict_recursively

    json_schema.setdefault("$id", schema_id)
    json_schema.setdefault("$schema", "http://json-schema.org/draft-07/schema#")
    json_schema = add_key_in_dict_recursively(
        json_schema,
        key="additionalProperties",
        value=False,
        criteria=lambda d: d.get("type") == "object" and "additionalProperties" not in d,
    )
    json_schema = resolve_refs(json_schema)
    if "$defs" in json_schema:
        del json_schema["$defs"]

    return json_schema


def resolve_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve all local $refs in the given JSON Schema using $defs as the source.

    Required since pydantic does not support nested schemas without $refs. https://github.com/pydantic/pydantic/issues/889

    Args:
         schema: A JSON Schema as a dictionary, which may contain "$refs" and "$defs".
    Returns:  A new dictionary with all local $refs resolved against $defs.
    """
    defs = schema.get("$defs", {})
    # Work on a deep copy so we don't mutate the original schema.
    schema_copy = deepcopy(schema)

    def _resolve(node: Any) -> Any:
        """Recursively walk through the node, resolving any local $refs to the corresponding definitions in 'defs'."""
        if isinstance(node, dict):
            # If this node is a reference to something in #/$defs/...
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                def_name = ref.replace("#/$defs/", "")
                if def_name in defs:
                    # Replace the entire node with the referenced definition.
                    return _resolve(deepcopy(defs[def_name]))
                else:
                    raise KeyError(f"Definition '{def_name}' not found in $defs.")

            # If not a ref, or doesn't start with #/$defs/, just walk deeper.
            return {key: _resolve(value) for key, value in node.items()}

        elif isinstance(node, list):
            # Recurse into each item of the list
            return [_resolve(item) for item in node]

        # For scalars (str, int, bool, None, etc.), return them as-is.
        return node

    return _resolve(schema_copy)
